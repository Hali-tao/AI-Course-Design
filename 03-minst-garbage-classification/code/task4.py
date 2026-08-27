import os
import time
import mindspore as ms
from mindspore import nn, context, dataset
from mindspore.train import Model
from mindspore.train.callback import LossMonitor
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV

# 1. 运行环境配置 (强制 CPU 模式)
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

GARBAGE_TRAIN_DIR = "./data_en/train/"
GARBAGE_TEST_DIR = "./data_en/test/"
PRETRAINED_CKPT = "./checkpoint/lenet_mnist-3_1875.ckpt"

# 2. 统一的高质量 3 通道彩色数据集构建 (严格控制变量)
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
# 模型 1: 改进前网络 (原生 3 通道 LeNet5FineTune)
# =====================================================================
class LeNet5FineTune(nn.Cell):
    def __init__(self, num_class=26):
        super(LeNet5FineTune, self).__init__()
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
# 模型 2: 改进方案 A (手工嵌入轻量残差连接块的 LeNet5Residual)
# =====================================================================
class ResidualBlock(nn.Cell):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, pad_mode='same')
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def construct(self, x):
        identity = x
        out = self.bn(self.conv(x))
        out = self.relu(out + identity)
        return out

class LeNet5Residual(nn.Cell):
    def __init__(self, num_class=26):
        super(LeNet5Residual, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, pad_mode='valid')
        self.bn1 = nn.BatchNorm2d(6)
        self.res1 = ResidualBlock(6)
        
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.bn2 = nn.BatchNorm2d(16)
        self.res2 = ResidualBlock(16)
        
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, num_class)
        
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
    
    def construct(self, x):
        x = self.max_pool2d(self.res1(self.relu(self.bn1(self.conv1(x)))))
        x = self.max_pool2d(self.res2(self.relu(self.bn2(self.conv2(x)))))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# =====================================================================
# 模型 3: 改进方案 B (纯原生底层组装 MobileNetV2 适配版 —— 彻底免导包错误)
# =====================================================================
class InvertedResidual(nn.Cell):
    """MobileNetV2 核心：倒残差短路模块"""
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(in_channels * expand_ratio)

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, stride=1, pad_mode='same', has_bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6()
            ])
        
        # 3x3 深度可分离卷积 (Depthwise Convolution)
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, pad_mode='same', group=hidden_dim, has_bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6()
        ])
        
        # 1x1 逐点卷积线性降维 (Linear Bottleneck)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, stride=1, pad_mode='same', has_bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.SequentialCell(layers)

    def construct(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class GarbageMobileNetV2(nn.Cell):
    """完全自主拓扑构建的标准 MobileNetV2，专为 32x32 图像优化分辨率传导"""
    def __init__(self, num_class=26):
        super(GarbageMobileNetV2, self).__init__()
        
        # (输入通道, 输出通道, 重复次数, 步长Stride, 升维倍数Expand)
        self.config = [
            [32,  16,  1, 1, 1], 
            [16,  24,  2, 1, 6], # Stride=1 保护初始小图尺寸不收缩
            [24,  32,  3, 2, 6], # 32x32 -> 16x16
            [32,  64,  4, 2, 6], # 16x16 -> 8x8
            [64,  96,  3, 1, 6], 
            [96,  160, 3, 2, 6], # 8x8 -> 4x4
            [160, 320, 1, 1, 6], 
        ]
        
        # 根基根节点卷积
        layers = [
            nn.Conv2d(3, 32, 3, stride=1, pad_mode='same', has_bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        ]
        
        # 遍历配置构建庞大的深层主干
        in_channels = 32
        for c, t, n, s, e in self.config:
            out_channels = t
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual(in_channels, out_channels, stride, expand_ratio=e))
                in_channels = out_channels
                
        # 尾部特征聚集层
        layers.extend([
            nn.Conv2d(in_channels, 1280, 1, stride=1, pad_mode='same', has_bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6()
        ])
        
        self.features = nn.SequentialCell(layers)
        
        # 26 类高维密集分类头
        self.classifier = nn.SequentialCell([
            nn.Dropout(p=0.2),
            nn.Dense(1280, num_class)
        ])

    def construct(self, x):
        x = self.features(x)
        x = nn.AvgPool2d(kernel_size=x.shape[2:])(x)
        x = ms.ops.reshape(x, (x.shape[0], x.shape[1]))
        x = self.classifier(x)
        return x

# =====================================================================
# 4. 智能权重注入模块
# =====================================================================
def load_filtered_checkpoint(net, model_type="lenet"):
    if os.path.exists(PRETRAINED_CKPT):
        param_dict = ms.load_checkpoint(PRETRAINED_CKPT)
        net_shapes = {name: param.shape for name, param in net.parameters_and_names()}
        filtered_dict = {}
        
        for k, v in param_dict.items():
            target_key = k
            if model_type == "residual" and k == "conv2.weight":
                target_key = "conv2.weight"
            
            if target_key in net_shapes and v.shape == net_shapes[target_key]:
                filtered_dict[target_key] = v
                
        ms.load_param_into_net(net, filtered_dict)

# =====================================================================
# 5. 自动化容错运行引擎
# =====================================================================
def execute_safe_run(model_type="lenet", target_epoch=50):
    ds_train = create_garbage_dataset(GARBAGE_TRAIN_DIR, batch_size=32, training=True)
    ds_test = create_garbage_dataset(GARBAGE_TEST_DIR, batch_size=32, training=False)
    
    if model_type == "lenet":
        print(f"\n▶▶▶ 队列(1/3)：【基准组 —— 原生彩色 LeNet-5 网络】 ◀◀◀")
        net = LeNet5FineTune(num_class=26)
        load_filtered_checkpoint(net, "lenet")
    elif model_type == "residual":
        print(f"\n▶▶▶ 队列(2/3)：【改进组A —— 魔改 LeNet-Residual 残差网络】 ◀◀◀")
        net = LeNet5Residual(num_class=26)
        load_filtered_checkpoint(net, "residual")
    else:
        print(f"\n▶▶▶ 队列(3/3)：【改进组B —— 现代深层 MobileNetV2 网络】 ◀◀◀")
        net = GarbageMobileNetV2(num_class=26)
    
    num_trainable = sum([p.size for p in net.get_parameters() if p.requires_grad])
    print(f"当前网络就绪，可训练参数总容量: {num_trainable} 个")
    
    # 严格保持完全一致的超参数
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)
    model = Model(net, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})
    
    start_time = time.time()
    
    # 核心高级防御区：捕获原生网络在50轮由于缺失保护导致的NaN梯度崩溃
    try:
        model.train(epoch=target_epoch, train_dataset=ds_train, 
                    callbacks=[LossMonitor(per_print_times=ds_train.get_dataset_size())], 
                    dataset_sink_mode=False)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        metrics = model.eval(ds_test, dataset_sink_mode=False)
        acc_result = f"{metrics['accuracy']:.4%}"
        
    except ValueError as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n⚠️ [防崩保护] 捕获当前模型在极端长线迭代中发生 NaN 溢出。系统已自动切出到下一队列。")
        acc_result = "崩溃(NaN溢出)"
        
    return elapsed_time, acc_result, num_trainable

if __name__ == "__main__":
    # 执行 50 次全量极限迭代压力测试
    TOTAL_EPOCHS = 50
    results = {}
    
    # 队列全自动化流转
    for mode in ["lenet", "residual", "mobilenet"]:
        t, acc, params = execute_safe_run(model_type=mode, target_epoch=TOTAL_EPOCHS)
        results[mode] = {"time": t, "acc": acc, "params": params}
        
    # =====================================================================
    # 6. 任务(4) 三代演进终极看板打印输出
    # =====================================================================
    print("\n" + "="*45 + f" 任务(4) 三种模型拓扑结构【50阶】终极对比看板 " + "="*45)
    print(f"{'网络拓扑状态':<25}\t{'可训练参数量':<12}\t{'设置迭代轮次':<12}\t{'总训练耗时':<12}\t{'测试集最终准确率'}")
    
    for k, v in results.items():
        name = "改进前 (原生彩色LeNet)" if k=="lenet" else "改进后A (魔改残差LeNet)" if k=="residual" else "改进后B (现代MobileNetV2)"
        print(f"{name:<25}\t{v['params']:<12}\t{f'{TOTAL_EPOCHS} Epoch':<12}\t{v['time']:<.2f}秒\t\t{v['acc']}")
    print("="*134)