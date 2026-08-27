import os
import time
import mindspore as ms
from mindspore import nn, context, dataset
from mindspore.train import Model
from mindspore.train.callback import LossMonitor
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV

# 环境配置
context.set_context(mode=context.GRAPH_MODE)
ms.set_device("CPU")

# 路径配置
GARBAGE_TRAIN_DIR = "./data_en/train/"
GARBAGE_TEST_DIR = "./data_en/test/"
PRETRAINED_CKPT = "./checkpoint/lenet_mnist-3_1875.ckpt"

# =====================================================================
# 1. 数据集构建
# =====================================================================
def create_garbage_dataset(data_path, batch_size=32, training=True):
    ds = dataset.ImageFolderDataset(data_path, shuffle=training)
    transform = [
        CV.Decode(), 
        CV.Resize((32, 32)),
        CV.Rescale(1.0 / 255.0, 0.0), 
        CV.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
        CV.HWC2CHW()
    ]
    type_cast_op = C.TypeCast(ms.int32)
    ds = ds.map(operations=transform, input_columns="image")
    ds = ds.map(operations=type_cast_op, input_columns="label")
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds

# =====================================================================
# 2. 基础 LeNet5 网络
# =====================================================================
class BaseLeNet5(nn.Cell):
    def __init__(self, num_class=26):
        super(BaseLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
    
    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# =====================================================================
# 3. 增强版 LoRA 适配层与 LoRA LeNet5
# =====================================================================
class LoRADense(nn.Cell):
    def __init__(self, in_channels, out_channels, r=4, alpha=8):
        super(LoRADense, self).__init__()
        self.original_dense = nn.Dense(in_channels, out_channels)
        self.original_dense.weight.requires_grad = False 
        # 学术修正：允许基础偏置项自由微调，以自适应跨域转换后的数据分布平移
        self.original_dense.bias.requires_grad = True 
        
        self.lora_A = ms.Parameter(ms.common.initializer.initializer('normal', [in_channels, r]), name="lora_A")
        self.lora_B = ms.Parameter(ms.common.initializer.initializer('zeros', [r, out_channels]), name="lora_B")
        # 核心修正：引入 alpha 常量放大 PEFT 低秩空间增量信号的权重力道
        self.scale = float(alpha) / r

    def construct(self, x):
        base_out = self.original_dense(x)
        lora_out = ms.ops.matmul(ms.ops.matmul(x, self.lora_A), self.lora_B) * self.scale
        return base_out + lora_out

class LoRALeNet5(nn.Cell):
    def __init__(self, num_class=26):
        super(LoRALeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = LoRADense(16 * 5 * 5, 120, r=4, alpha=8)
        self.fc2 = LoRADense(120, 84, r=4, alpha=8)
        self.fc3 = nn.Dense(84, num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# =====================================================================
# 4. 分支动态权重加载
# =====================================================================
def load_and_filter_weights(net):
    if os.path.exists(PRETRAINED_CKPT):
        param_dict = ms.load_checkpoint(PRETRAINED_CKPT)
        net_param_shapes = {name: param.shape for name, param in net.parameters_and_names()}
        filtered_dict = {}
        
        is_lora_net = any("original_dense" in name for name in net_param_shapes.keys())
        net_type_label = "LoRA网络" if is_lora_net else "基础网络"
        
        print("\n" + "-"*15 + f" 预训练权重热加载日志 ({net_type_label}) " + "-"*15)
        
        for k, v in param_dict.items():
            if any(x in k for x in ["moments", "global_step", "learning_rate", "momentum"]):
                continue
            
            target_key = k
            if is_lora_net:
                if "fc1" in k and "original_dense" not in k:
                    target_key = k.replace("fc1", "fc1.original_dense")
                elif "fc2" in k and "original_dense" not in k:
                    target_key = k.replace("fc2", "fc2.original_dense")
            else:
                target_key = k
            
            if "conv1.weight" in k and net_param_shapes.get(target_key) == (6, 3, 5, 5):
                v_expanded = ms.ops.tile(v, (1, 3, 1, 1)) / 3.0
                filtered_dict[target_key] = ms.Parameter(v_expanded, name=target_key)
                print(f"【成功适配】{k} (6,1,5,5) -> 扩展均分适配为 3 通道 (6,3,5,5)")
                continue
            
            if target_key in net_param_shapes:
                if v.shape == net_param_shapes[target_key]:
                    filtered_dict[target_key] = v
                    print(f"【成功加载】{k} -> {target_key}")
                else:
                    print(f"【拦截跳过】{k} 发生形状冲突：原形状 {v.shape} ≠ 期待形状 {net_param_shapes[target_key]}")
                    
        ms.load_param_into_net(net, filtered_dict)
        print("-" * 62 + "\n")
    else:
        print(f"【警告】未找到权重文件: {PRETRAINED_CKPT}")

# =====================================================================
# 5. 核心对比中心
# =====================================================================
def execute_experiment(strategy="full"):
    print(f"\n▶▶▶ 正在启动策略: 【{strategy.upper()} 微调】 ◀◀◀")
    ds_train = create_garbage_dataset(GARBAGE_TRAIN_DIR, batch_size=32, training=True)
    ds_test = create_garbage_dataset(GARBAGE_TEST_DIR, batch_size=32, training=False)
    
    net = LoRALeNet5(num_class=26) if strategy == "lora" else BaseLeNet5(num_class=26)
    load_and_filter_weights(net)
    
    # 冻结微调修正：仅冻结全连接层的 weight 矩阵，允许其原生的 bias 更新以对齐新任务偏置
    if strategy == "freeze":
        for param in net.get_parameters():
            if ("fc1" in param.name or "fc2" in param.name) and "weight" in param.name:
                param.requires_grad = False
                
    trainable_params = filter(lambda p: p.requires_grad, net.get_parameters())
    num_trainable = sum([p.size for p in net.get_parameters() if p.requires_grad])
    print(f"当前模式下实际可训练参数量: {num_trainable} 个")
    
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    
    # 【核心修正】：重新精细化编排控制变量学习率。
    # LoRA 需要更大的冲击力（0.01）来克服零初始化滞后；Freeze 放开 bias 匹配 0.002 稳步迭代
    if strategy == "lora":
        lr = 0.01
    elif strategy == "freeze":
        lr = 0.01
    else:
        lr = 0.002 # Full 
        
    optimizer = nn.Momentum(trainable_params, learning_rate=lr, momentum=0.9)
    model = Model(net, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})
    
    if strategy == "lora":
        EPOCHS = 50
    elif strategy == "freeze":
        EPOCHS = 50
    else:
        EPOCHS = 25 # Full 

    start_time = time.time()
    model.train(epoch=EPOCHS, train_dataset=ds_train, callbacks=[LossMonitor(per_print_times=ds_train.get_dataset_size())], dataset_sink_mode=False)
    end_time = time.time()
    
    metrics = model.eval(ds_test, dataset_sink_mode=False)
    return end_time - start_time, metrics['accuracy'], num_trainable

if __name__ == "__main__":
    results = {}
    for mode in ["full", "freeze", "lora"]:
        t, acc, params = execute_experiment(strategy=mode)
        results[mode] = {"time": t, "acc": acc, "params": params}
        
    print("\n" + "="*43 + " 任务(3) 实验指标横向对比看板(优化进阶版) " + "="*43)
    print(f"{'微调策略':<12}\t{'可训练参数量':<12}\t{'总训练耗时(25 Epoch)':<18}\t{'测试集最终准确率':<15}")
    for k, v in results.items():
        strategy_name = "全量微调" if k=="full" else "冻结微调" if k=="freeze" else "LoRA微调"
        print(f"{strategy_name:<12}\t{v['params']:<12}\t{v['time']:<.2f}秒\t\t\t{v['acc']:.4%}")
    print("=" * 122)