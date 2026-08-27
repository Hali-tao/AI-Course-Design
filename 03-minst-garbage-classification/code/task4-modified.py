import os
import time
import numpy as np
import matplotlib.pyplot as plt
import mindspore as ms
from mindspore import nn, context, dataset
from mindspore.train import Model
from mindspore.train.callback import LossMonitor, TimeMonitor
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV

# =====================================================================
# 步骤 0: 计算硬件与环境配置
# =====================================================================
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

GARBAGE_TRAIN_DIR = "./data_en/train/"
GARBAGE_TEST_DIR = "./data_en/test/"
PRETRAINED_CKPT = "./checkpoint/lenet_mnist-3_1875.ckpt"

# =====================================================================
# 步骤 1: 统一的高质量 3 通道彩色数据集构建 (严格控制变量)
# =====================================================================
def create_garbage_dataset(data_path, batch_size=32, training=True):
    ds = dataset.ImageFolderDataset(data_path, num_parallel_workers=4, shuffle=training)
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
        return self.relu(out + identity)

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
# 模型 3: 改进方案 B (纯原生底层组装 MobileNetV2 跨模态适配版)
# =====================================================================
class InvertedResidual(nn.Cell):
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
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, pad_mode='same', group=hidden_dim, has_bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(),
            nn.Conv2d(hidden_dim, out_channels, 1, stride=1, pad_mode='same', has_bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        self.conv = nn.SequentialCell(layers)

    def construct(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class GarbageMobileNetV2(nn.Cell):
    def __init__(self, num_class=26):
        super(GarbageMobileNetV2, self).__init__()
        self.config = [
            [32,  16,  1, 1, 1], 
            [16,  24,  2, 1, 6], 
            [24,  32,  3, 2, 6], 
            [32,  64,  4, 2, 6], 
            [64,  96,  3, 1, 6], 
            [96,  160, 3, 2, 6], 
            [160, 320, 1, 1, 6], 
        ]
        
        layers = [
            nn.Conv2d(3, 32, 3, stride=1, pad_mode='same', has_bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        ]
        
        in_channels = 32
        for c, t, n, s, e in self.config:
            out_channels = t
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual(in_channels, out_channels, stride, expand_ratio=e))
                in_channels = out_channels
                
        layers.extend([
            nn.Conv2d(in_channels, 1280, 1, stride=1, pad_mode='same', has_bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6()
        ])
        self.features = nn.SequentialCell(layers)
        self.classifier = nn.SequentialCell([
            nn.Dropout(p=0.2),
            nn.Dense(1280, num_class)
        ])

    def construct(self, x):
        x = self.features(x)
        x = nn.AvgPool2d(kernel_size=x.shape[2:])(x)
        x = ms.ops.reshape(x, (x.shape[0], x.shape[1]))
        return self.classifier(x)

# =====================================================================
# 步骤 2: 严格安全检查的预训练参数注入
# =====================================================================
def load_filtered_checkpoint(net, model_type="lenet"):
    if os.path.exists(PRETRAINED_CKPT):
        param_dict = ms.load_checkpoint(PRETRAINED_CKPT)
        net_shapes = {name: param.shape for name, param in net.parameters_and_names()}
        filtered_dict = {}
        
        for k, v in param_dict.items():
            target_key = k
            if "fc3" not in k and "conv1" not in k:  # 锁死首尾冲突层
                if target_key in net_shapes and v.shape == net_shapes[target_key]:
                    filtered_dict[target_key] = v
        ms.load_param_into_net(net, filtered_dict)

# =====================================================================
# 步骤 3: 自动化统一调度执行引擎
# =====================================================================
def execute_safe_run(model_type="lenet", target_epoch=30):
    ds_train = create_garbage_dataset(GARBAGE_TRAIN_DIR, batch_size=32, training=True)
    ds_test = create_garbage_dataset(GARBAGE_TEST_DIR, batch_size=32, training=False)
    
    steps_per_epoch = ds_train.get_dataset_size()
    total_steps = steps_per_epoch * target_epoch
    
    if model_type == "lenet":
        print(f"\n▶▶▶ 队列(1/3)：【基准组 —— 原生彩色 LeNet-5 + 动态学习率】 ◀◀◀")
        net = LeNet5FineTune(num_class=26)
        load_filtered_checkpoint(net, "lenet")
    elif model_type == "residual":
        print(f"\n▶▶▶ 队列(2/3)：【改进组A —— 魔改 LeNet-Residual + 动态学习率】 ◀◀◀")
        net = LeNet5Residual(num_class=26)
        load_filtered_checkpoint(net, "residual")
    else:
        print(f"\n▶▶▶ 队列(3/3)：【改进组B —— 现代深层 MobileNetV2 + 动态学习率】 ◀◀◀")
        net = GarbageMobileNetV2(num_class=26)

    num_trainable = sum([p.size for p in net.get_parameters() if p.requires_grad])
    print(f"当前网络就绪，可训练参数总容量: {num_trainable} 个")
    
    # ⭐ 严格控制变量：注入完全一致的四段式常数退火动态学习率
    milestone = [
        steps_per_epoch * 8,   # 1-8 轮：0.005 高速知识对齐
        steps_per_epoch * 16,  # 9-16 轮：0.001 深度破局俯冲
        steps_per_epoch * 24,  # 17-24 轮：0.0002 边界精细微雕
        total_steps            # 25-30 轮：0.00005 极限安全降落
    ]
    learning_rates = [0.005, 0.001, 0.0002, 0.00005]
    dynamic_lr = nn.piecewise_constant_lr(milestone, learning_rates)
    
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Momentum(net.trainable_params(), learning_rate=dynamic_lr, momentum=0.9)
    model = Model(net, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})
    
    start_time = time.time()
    try:
        model.train(epoch=target_epoch, train_dataset=ds_train, 
                    callbacks=[LossMonitor(per_print_times=steps_per_epoch), TimeMonitor(data_size=steps_per_epoch)], 
                    dataset_sink_mode=False)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        metrics = model.eval(ds_test, dataset_sink_mode=False)
        acc_val = metrics['accuracy']
        acc_result_str = f"{acc_val:.4%}"
        
    except ValueError:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n⚠️ 捕获当前模型发生非对称计算异常。")
        acc_val = 0.0
        acc_result_str = "崩溃(NaN溢出)"
        
    return elapsed_time, acc_val, acc_result_str, num_trainable

# =====================================================================
# 步骤 4: 学术级学术规范双 Y 轴柱状图自动化绘制函数
# =====================================================================
def draw_performance_chart(results):
    models = ['LeNet-5\n(Baseline)', 'LeNet-Residual\n(Ours-A)', 'MobileNetV2\n(Ours-B)']
    accuracies = [results['lenet']['acc_val'] * 100, results['residual']['acc_val'] * 100, results['mobilenet']['acc_val'] * 100]
    times = [results['lenet']['time'], results['residual']['time'], results['mobilenet']['time']]

    x = np.arange(len(models))
    width = 0.35  

    fig, ax1 = plt.subplots(figsize=(9, 6), dpi=300)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
    plt.rcParams['axes.unicode_minus'] = False     

    # 绘制左轴：测试集准确率
    color_acc = '#1f77b4'
    rects1 = ax1.bar(x - width/2, accuracies, width, label='测试集准确率 (%)', 
                     color=color_acc, alpha=0.85, edgecolor='black', linewidth=0.8)
    ax1.set_xlabel('网络拓扑架构演进', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel('最终测试集准确率 (%)', color=color_acc, fontsize=12, fontweight='bold')
    ax1.set_yticks(np.arange(0, 101, 10))
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)

    # 绘制右轴：总训练耗时
    ax2 = ax1.twinx()  
    color_time = '#e34a33'
    rects2 = ax2.bar(x + width/2, times, width, label='30轮训练总耗时 (秒)', 
                     color=color_time, alpha=0.85, edgecolor='black', linewidth=0.8)
    ax2.set_ylabel('总训练耗时 (秒, CPU环境)', color=color_time, fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_time)

    # 自动数值标注内部闭包函数
    def autolabel(rects, ax, is_time=False):
        for rect in rects:
            height = rect.get_height()
            if is_time:
                label_text = f'{height:.1f}s' if height < 60 else f'{height/60:.1f}min'
            else:
                label_text = f'{height:.2f}%' if height > 0 else "N/A"
            ax.annotate(label_text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1, ax1, is_time=False)
    autolabel(rects2, ax2, is_time=True)

    plt.xticks(x, models, fontsize=11, fontweight='bold')
    plt.title('任务(4) 统一动态学习率下三代模型性能与计算成本横向对比看板', fontsize=13, fontweight='bold', pad=15)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    fig.tight_layout()
    chart_save_path = './model_performance_comparison.png'
    plt.savefig(chart_save_path, dpi=300)
    print(f"\n🎉 [成功] 任务(4) 模型三代演进横向对比柱状图已成功导出至: {chart_save_path}")
    plt.close()

# =====================================================================
# 主入口：全自动化流水线触发
# =====================================================================
if __name__ == "__main__":
    TOTAL_EPOCHS = 30  
    raw_results = {}
    
    # 依次触发三代拓扑网络测试
    for mode in ["lenet", "residual", "mobilenet"]:
        t, acc_val, acc_str, params = execute_safe_run(model_type=mode, target_epoch=TOTAL_EPOCHS)
        raw_results[mode] = {"time": t, "acc_val": acc_val, "acc_str": acc_str, "params": params}
        
    # 1. 打印终端字符看板
    print("\n" + "="*39 + f" 任务(4) 动态学习率控制组【30 Epoch】终极对比看板 " + "="*39)
    print(f"{'网络拓扑状态':<25}\t{'可训练参数量':<12}\t{'全局学习率策略':<16}\t{'总训练耗时':<12}\t{'测试集最终准确率'}")
    
    for k, v in raw_results.items():
        name = "改进前 (原生彩色LeNet)" if k=="lenet" else "改进后A (魔改残差LeNet)" if k=="residual" else "改进后B (现代MobileNetV2)"
        print(f"{name:<25}\t{v['params']:<12}\t{'Piecewise-LR':<16}\t{v['time']:<.2f}秒\t\t{v['acc_str']}")
    print("="*122)

    # 2. 自动化调用绘图引擎出图
    draw_performance_chart(raw_results)