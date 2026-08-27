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
# 1. 数据集构建（升级为 3 通道彩色流）
# =====================================================================
def create_garbage_dataset(data_path, batch_size=32, training=True):
    ds = dataset.ImageFolderDataset(data_path, shuffle=training)
    transform = [
        CV.Decode(), 
        CV.Resize((32, 32)),
        CV.Rescale(1.0 / 255.0, 0.0), 
        # 升级点：保留彩色，采用通用的三通道标准均值与标准差
        CV.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
        CV.HWC2CHW()
    ]
    type_cast_op = C.TypeCast(ms.int32)
    ds = ds.map(operations=transform, input_columns="image")
    ds = ds.map(operations=type_cast_op, input_columns="label")
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds

# =====================================================================
# 2. 基础 LeNet5 网络（输入通道升级为 3 通道）
# =====================================================================
class BaseLeNet5(nn.Cell):
    def __init__(self, num_class=26):
        super(BaseLeNet5, self).__init__()
        # 核心改动：输入通道数由 1 改为 3
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
# 3. LoRA 适配层与 3 通道 LoRA 版 LeNet5 网络
# =====================================================================
class LoRADense(nn.Cell):
    """自定义实现的低秩自适应全连接层"""
    def __init__(self, in_channels, out_channels, r=4):
        super(LoRADense, self).__init__()
        self.original_dense = nn.Dense(in_channels, out_channels)
        self.original_dense.weight.requires_grad = False # 彻底冻结原生稠密权重矩阵
        
        # 引入低秩分解旁路 A 和 B (仅训练这部分参数)
        self.lora_A = ms.Parameter(ms.common.initializer.initializer('normal', [in_channels, r]), name="lora_A")
        self.lora_B = ms.Parameter(ms.common.initializer.initializer('zeros', [r, out_channels]), name="lora_B")
        self.scale = 1.0 / r

    def construct(self, x):
        base_out = self.original_dense(x)
        lora_out = ms.ops.matmul(ms.ops.matmul(x, self.lora_A), self.lora_B) * self.scale
        return base_out + lora_out

class LoRALeNet5(nn.Cell):
    """挂载 LoRA 且支持 3 通道的网络拓扑"""
    def __init__(self, num_class=26):
        super(LoRALeNet5, self).__init__()
        # 改动点：输入调整为 3 通道
        self.conv1 = nn.Conv2d(3, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        
        self.fc1 = LoRADense(16 * 5 * 5, 120, r=4)
        self.fc2 = LoRADense(120, 84, r=4)
        self.fc3 = nn.Dense(84, num_class) # 分类头正常训练
        
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        
        # LoRA模式下：强制锁死两个基础卷积层
        self.conv1.weight.requires_grad = False
        self.conv2.weight.requires_grad = False

    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# =====================================================================
# 4. 严格形状安全比对的权重加载逻辑
# =====================================================================
def load_and_filter_weights(net):
    if os.path.exists(PRETRAINED_CKPT):
        param_dict = ms.load_checkpoint(PRETRAINED_CKPT)
        
        # 获取新网络当前的参数结构图
        net_param_shapes = {name: param.shape for name, param in net.parameters_and_names()}
        filtered_dict = {}
        
        for k, v in param_dict.items():
            if "moments" in k or "global_step" in k or "learning_rate" in k or "momentum" in k:
                continue
            
            # 为 LoRA 层进行名称映射对齐
            target_key = k
            if "fc1" in k and "original_dense" not in k:
                target_key = k.replace("fc1", "fc1.original_dense")
            elif "fc2" in k and "original_dense" not in k:
                target_key = k.replace("fc2", "fc2.original_dense")
            
            # 严格安全守卫：必须在新模型中存在，且矩阵形状完全一致才允许加载
            if target_key in net_param_shapes and v.shape == net_param_shapes[target_key]:
                filtered_dict[target_key] = v
                
        ms.load_param_into_net(net, filtered_dict)

# =====================================================================
# 5. 核心对比调度中心
# =====================================================================
def execute_experiment(strategy="full"):
    print(f"\n▶▶▶ 正在启动策略: 【{strategy.upper()} 微调】 ◀◀◀")
    ds_train = create_garbage_dataset(GARBAGE_TRAIN_DIR, batch_size=32, training=True)
    ds_test = create_garbage_dataset(GARBAGE_TEST_DIR, batch_size=32, training=False)
    
    if strategy == "lora":
        net = LoRALeNet5(num_class=26)
    else:
        net = BaseLeNet5(num_class=26)
    
    # 注入预训练参数
    load_and_filter_weights(net)
    
    # 核心修正：精准遍历参数列表，实现真正、彻底的冻结控制
    if strategy == "freeze":
        for param in net.get_parameters():
            if "fc3" not in param.name:
                param.requires_grad = False  # 锁死除末端分类头 fc3 之外的所有层
                
    # 重新过滤提取当前模式激活的可训练参数
    trainable_params = filter(lambda p: p.requires_grad, net.get_parameters())
    num_trainable = sum([p.size for p in net.get_parameters() if p.requires_grad])
    print(f"当前模式下实际可训练参数量: {num_trainable} 个")
    
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Momentum(trainable_params, learning_rate=0.005, momentum=0.9)
    model = Model(net, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})
    
    # 计时开始
    start_time = time.time()
    model.train(epoch=25, train_dataset=ds_train, callbacks=[LossMonitor(per_print_times=ds_train.get_dataset_size())], dataset_sink_mode=False)
    end_time = time.time()
    
    # 独立测试集验证表现
    metrics = model.eval(ds_test, dataset_sink_mode=False)
    elapsed_time = end_time - start_time
    
    return elapsed_time, metrics['accuracy'], num_trainable

if __name__ == "__main__":
    results = {}
    # 顺序轮流触发三个微调策略
    for mode in ["full", "freeze", "lora"]:
        t, acc, params = execute_experiment(strategy=mode)
        results[mode] = {"time": t, "acc": acc, "params": params}
        
    # =====================================================================
    # 6. 重构后全新对比看板输出
    # =====================================================================
    print("\n" + "="*50 + " 任务(3) 实验指标横向对比看板(RGB重构版) " + "="*50)
    print(f"{'微调策略':<12}\t{'可训练参数量':<12}\t{'总训练耗时(3 Epoch)':<18}\t{'测试集最高准确率':<15}")
    for k, v in results.items():
        strategy_name = "全量微调" if k=="full" else "冻结微调" if k=="freeze" else "LoRA微调"
        print(f"{strategy_name:<12}\t{v['params']:<12}\t{v['time']:<.2f}秒\t\t\t{v['acc']:.4%}")
    print("="*129)