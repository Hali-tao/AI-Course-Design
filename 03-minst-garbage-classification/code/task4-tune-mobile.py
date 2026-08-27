import os
import sys
import time
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops as P
from mindspore import Tensor, set_context
from mindspore import load_checkpoint, load_param_into_net
import mindspore.dataset as de
import mindspore.dataset.vision as C
import mindspore.dataset.transforms as C2

# =====================================================================
# [系统配置] 保持动态图高性能
# =====================================================================
set_context(mode=ms.PYNATIVE_MODE, device_target="GPU") 
os.environ['GLOG_v'] = '3'

CONFIG = {
    "num_classes": 26,
    "image_height": 224,
    "image_width": 224,
    "batch_size": 64,            # V100 大显存配置
    "eval_batch_size": 32,
    "local_dir": "./data_en",       
    "local_ckpt": "./mobilenet_v2.ckpt"  
}

# =====================================================================
# [数据加载器] 保持 4 线程并行驱动
# =====================================================================
def create_dataset(dataset_path, training=True, batch_size=64):
    data_path = os.path.join(dataset_path, 'train' if training else 'test')
    if not os.path.exists(data_path):
        data_path = os.path.join(dataset_path, 'data_en', 'train' if training else 'test')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"【错误】无法定位到数据路径：{data_path}")
        
    index_en = {'Seashell': 0, 'Lighter': 1, 'Old Mirror': 2, 'Broom': 3, 'Ceramic Bowl': 4, 'Toothbrush': 5, 'Disposable Chopsticks': 6, 'Dirty Cloth': 7,
                'Newspaper': 8, 'Glassware': 9, 'Basketball': 10, 'Plastic Bottle': 11, 'Cardboard': 12, 'Glass Bottle': 13, 'Metalware': 14, 'Hats': 15, 'Cans': 16, 'Paper': 17,
                'Vegetable Leaf': 18, 'Orange Peel': 19, 'Eggshell': 20, 'Banana Peel': 21, 'Battery': 22, 'Tablet capsules': 23, 'Fluorescent lamp': 24, 'Paint bucket': 25}
    
    ds = de.ImageFolderDataset(data_path, num_parallel_workers=4, class_indexing=index_en)
    normalize_op = C.Normalize(mean=[0.485*255, 0.456*255, 0.406*255], std=[0.229*255, 0.224*255, 0.225*255])
    change_swap_op = C.HWC2CHW()
    type_cast_op = C2.TypeCast(ms.int32)

    if training:
        crop_decode_resize = C.RandomCropDecodeResize(CONFIG["image_height"], scale=(0.08, 1.0), ratio=(0.75, 1.333))
        horizontal_flip_op = C.RandomHorizontalFlip(prob=0.5)
        train_trans = [crop_decode_resize, horizontal_flip_op, normalize_op, change_swap_op]
        ds = ds.map(input_columns="image", operations=train_trans, num_parallel_workers=4)
        ds = ds.map(input_columns="label", operations=type_cast_op, num_parallel_workers=4)
        ds = ds.shuffle(buffer_size=200)
        ds = ds.batch(batch_size, drop_remainder=True)
    else:
        decode_op = C.Decode()
        resize_op = C.Resize((256, 256))
        center_crop = C.CenterCrop(CONFIG["image_width"])
        eval_trans = [decode_op, resize_op, center_crop, normalize_op, change_swap_op]
        ds = ds.map(input_columns="image", operations=eval_trans, num_parallel_workers=4)
        ds = ds.map(input_columns="label", operations=type_cast_op, num_parallel_workers=4)
        ds = ds.batch(CONFIG["eval_batch_size"], drop_remainder=True)
    return ds

# =====================================================================
# [网络模型结构定义] (含 LoRA 算子)
# =====================================================================
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class GlobalAvgPooling(nn.Cell):
    def __init__(self): super().__init__()
    def construct(self, x): return P.mean(x, (2, 3))

class ConvBNReLU(nn.Cell):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode='pad', padding=padding, group=in_planes if groups != 1 else 1)
        self.features = nn.SequentialCell([conv, nn.BatchNorm2d(out_planes), nn.ReLU6()])
    def construct(self, x): return self.features(x)

class InvertedResidual(nn.Cell):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, has_bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.SequentialCell(layers)
    def construct(self, x):
        return P.add(x, self.conv(x)) if self.use_res_connect else self.conv(x)

class MobileNetV2Backbone(nn.Cell):
    def __init__(self, width_mult=1.):
        super().__init__()
        cfgs = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        input_channel = _make_divisible(32 * width_mult, 8)
        self.out_channels = _make_divisible(1280 * max(1.0, width_mult), 8)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        for t, c, n, s in cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                features.append(InvertedResidual(input_channel, output_channel, s if i == 0 else 1, expand_ratio=t))
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, self.out_channels, kernel_size=1))
        self.features = nn.SequentialCell(features)
    def construct(self, x): return self.features(x)

class LoRADense(nn.Cell):
    def __init__(self, in_features, out_features, r=4, alpha=8):
        super().__init__()
        self.dense = nn.Dense(in_features, out_features, has_bias=True)
        self.lora_A = ms.Parameter(Tensor(np.random.normal(0, 0.01, (r, in_features)).astype(np.float32)), name="lora_A")
        self.lora_B = ms.Parameter(Tensor(np.zeros((out_features, r)).astype(np.float32)), name="lora_B")
        self.scale = alpha / r
    def construct(self, x):
        base_output = self.dense(x)
        lora_output = P.matmul(x, self.lora_A.T)
        lora_output = P.matmul(lora_output, self.lora_B.T)
        return base_output + lora_output * self.scale

class MobileNetV2Head(nn.Cell):
    def __init__(self, input_channel=1280, num_classes=1000, is_lora=False):
        super().__init__()
        core_dense = LoRADense(input_channel, num_classes) if is_lora else nn.Dense(input_channel, num_classes, has_bias=True)
        self.head = nn.SequentialCell([GlobalAvgPooling(), core_dense])
    def construct(self, x): return self.head(x)

class MobileNetV2(nn.Cell):
    def __init__(self, num_classes=1000, is_lora=False):
        super().__init__()
        self.backbone = MobileNetV2Backbone()
        self.head = MobileNetV2Head(input_channel=self.backbone.out_channels, num_classes=num_classes, is_lora=is_lora)
    def construct(self, x): return self.head(self.backbone(x))

def load_and_align_weights(net):
    raw_dict = load_checkpoint(CONFIG["local_ckpt"])
    aligned_dict = {}
    for key, param in raw_dict.items():
        if "head" in key or "classifier" in key: continue
        new_key = "backbone." + key if not key.startswith("backbone.") else key
        aligned_dict[new_key] = param
    load_param_into_net(net, aligned_dict)

# =====================================================================
# [核心动态学习率生成器] 完美移植 6.1 节设计
# =====================================================================
def get_dynamic_lr(steps_per_epoch):
    total_steps = steps_per_epoch * 30
    lr_each_step = []
    for step in range(total_steps):
        if step < steps_per_epoch * 8:
            lr_each_step.append(0.005)   # 1-8 轮：高速知识对齐
        elif step < steps_per_epoch * 16:
            lr_each_step.append(0.001)   # 9-16 轮：深度破局俯冲
        elif step < steps_per_epoch * 24:
            lr_each_step.append(0.0002)  # 17-24 轮：边界精细微雕
        else:
            lr_each_step.append(0.00005) # 25-30 轮：极限安全降落
    return Tensor(lr_each_step, ms.float32)

# =====================================================================
# [多维实验中枢]
# =====================================================================
def run_experiment(group_name, mode, max_epochs, use_dynamic_lr, train_ds, test_ds):
    steps_per_epoch = train_ds.get_dataset_size()
    
    # 1. 初始化网络与策略梯度控制
    net = MobileNetV2(num_classes=CONFIG["num_classes"], is_lora=(mode == "LoRA"))
    load_and_align_weights(net)
    
    if mode == "Full-FT":
        for param in net.get_parameters(): param.requires_grad = True
    elif mode == "Freeze":
        for param in net.get_parameters(): param.requires_grad = False if "backbone" in param.name else True
    elif mode == "LoRA":
        for param in net.get_parameters():
            if "lora_A" in param.name or "lora_B" in param.name or "head.head.1.dense" in param.name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
    trainable_params = list(filter(lambda p: p.requires_grad, net.trainable_params()))
    trainable_weight_count = sum([np.prod(p.shape) for p in trainable_params])
    
    # 2. 注入学习率策略
    if use_dynamic_lr:
        lr = get_dynamic_lr(steps_per_epoch)
    else:
        lr = 0.005 # 固定学习率基准
        
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Momentum(params=trainable_params, learning_rate=lr, momentum=0.9)
    
    loss_net = nn.WithLossCell(net, loss_fn)
    train_net = nn.TrainOneStepCell(loss_net, optimizer)
    train_net.set_train(True)
    
    # 3. 执行滚轮训练
    start_time = time.time()
    last_loss = 0.0
    
    for epoch in range(max_epochs):
        step_idx = 0
        for data in train_ds.create_tuple_iterator():
            images, labels = data
            loss = train_net(images, labels)
            last_loss = loss.asnumpy()
            step_idx += 1
            
        # 为了防刷屏，每轮末尾打印一次宏观收敛即可
        print(f"      -> [组别:{group_name} | {mode}] Epoch {epoch+1}/{max_epochs} 完成 | 本轮末端 Loss: {last_loss:.4f}")
                
    elapsed_time = time.time() - start_time
    
    # 4. 全量验证集精度评测
    train_net.set_train(False)
    net.set_train(False)
    correct, total = 0, 0
    for data in test_ds.create_tuple_iterator():
        images, labels = data
        outputs = net(images)
        preds = np.argmax(outputs.asnumpy(), axis=1)
        correct += np.sum(preds == labels.asnumpy())
        total += labels.shape[0]
    final_acc = (correct / total) * 100 if total > 0 else 0.0
    
    return trainable_weight_count, elapsed_time, last_loss, final_acc

# =====================================================================
# [主运行入口]
# =====================================================================
if __name__ == "__main__":
    print("="*60 + "\n【交叉消融实验启动】加载基础数据集中...\n" + "="*60)
    train_dataset = create_dataset(CONFIG["local_dir"], training=True)
    test_dataset = create_dataset(CONFIG["local_dir"], training=False)
    
    # 定义 3 个大组的设计需求
    experimental_groups = [
        {"name": "1、50 Epoch 极限压力组", "epochs": 50, "dynamic_lr": False},
        {"name": "2、30 Epoch 早停性价比组", "epochs": 30, "dynamic_lr": False},
        {"name": "3、30 Epoch 动态学习率组", "epochs": 30, "dynamic_lr": True}
    ]
    modes = ["Full-FT", "Freeze", "LoRA"]
    
    # 结果收集器
    all_results = {}
    
    for group in experimental_groups:
        print(f"\n🚀 开始执行大组任务: 🌟 {group['name']} 🌟")
        all_results[group['name']] = {}
        for mode in modes:
            print(f"   执行策略: 【{mode}】...")
            p_count, t_cost, l_loss, f_acc = run_experiment(
                group_name=group['name'],
                mode=mode,
                max_epochs=group['epochs'],
                use_dynamic_lr=group['dynamic_lr'],
                train_ds=train_dataset,
                test_ds=test_dataset
            )
            all_results[group['name']][mode] = {
                "params": f"{p_count:,}",
                "time": f"{t_cost:.1f} 秒",
                "loss": f"{l_loss:.4f}",
                "acc": f"{f_acc:.2f}%"
            }

    # =====================================================================
    # [华丽落幕] 自动化跨维度综合大看板
    # =====================================================================
    print("\n" + "="*23 + " 6.2 节完备跨维度交叉成果看板 (V100 GPU) " + "="*23)
    for group_name, modes_data in all_results.items():
        print(f"\n📊 实验组别名称: {group_name}")
        print(f"| {'微调策略模式':<12} | {'参与反传参数量':<14} | {'总训练耗时':<12} | {'收尾阶段 Loss':<14} | {'测试集最终准确率':<14} |")
        print("|" + "-"*16 + "|" + "-"*16 + "|" + "-"*14 + "|" + "-"*18 + "|" + "-"*18 + "|")
        for mode, m in modes_data.items():
            print(f"| {mode:<14} | {m['params']:<14} | {m['time']:<12} | {m['loss']:<16} | {m['acc']:<16} |")
        print("-" * 80)