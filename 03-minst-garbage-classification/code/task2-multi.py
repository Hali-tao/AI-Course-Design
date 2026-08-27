import os
import mindspore as ms
from mindspore import nn, context, dataset
from mindspore.train import Model
from mindspore.train.callback import LossMonitor, TimeMonitor
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV

# =====================================================================
# 步骤 0: 环境与计算硬件配置（租了算力直接开启 GPU）
# =====================================================================
context.set_context(mode=context.GRAPH_MODE)
ms.set_device("CPU")  # ⭐ 已切换为 GPU 模式

# 数据集路径配置
GARBAGE_TRAIN_DIR = "./data_en/train/"   
GARBAGE_TEST_DIR = "./data_en/test/"     

# =====================================================================
# 步骤 1: 加载与优化 26 类垃圾分类数据集（100% 采用你给出的标准彩色流水线）
# =====================================================================
def create_garbage_dataset(data_path, batch_size=32, training=True):
    ds = dataset.ImageFolderDataset(data_path, num_parallel_workers=4, shuffle=training)
    
    # ⭐ 完美保留你给出的 3 通道高质量彩色增强流水线
    transform = [
        CV.Decode(),                                                         # 解码图片
        CV.Resize((32, 32)),                                                 # 缩放到 LeNet5 输入尺寸 (32x32)
        CV.Rescale(1.0 / 255.0, 0.0),                                        # 归一化到 [0, 1]
        CV.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # 3通道标准标准化
        CV.HWC2CHW()                                                         # 转换为 MindSpore 要求的 CHW 格式
    ]
    
    type_cast_op = C.TypeCast(ms.int32)        
    
    ds = ds.map(operations=transform, input_columns="image")
    ds = ds.map(operations=type_cast_op, input_columns="label")
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds

# =====================================================================
# 步骤 2: 初始化垃圾分类网络模型 (第一层适配 3 通道彩色输入)
# =====================================================================
class LeNet5ForGarbage(nn.Cell):
    def __init__(self, num_class=26):
        super(LeNet5ForGarbage, self).__init__()
        # 核心改动：将输入通道由 1 改为 3。让第一层卷积承担 RGB 颜色特征融合的任务
        self.conv1 = nn.Conv2d(3, 6, 5, pad_mode='valid')
        
        # 后续所有经典拓扑层和尺寸完全保持纯正的 LeNet-5 结构
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, num_class) # 输出修改为 26 类
        
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
    
    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# =====================================================================
# 主入口：微调训练 + 独立测试
# =====================================================================
def run_finetune_and_test():
    print("====== 步骤 1: 加载 26 类纯彩色垃圾分类数据集 ======")
    ds_train = create_garbage_dataset(GARBAGE_TRAIN_DIR, batch_size=32, training=True)
    ds_test = create_garbage_dataset(GARBAGE_TEST_DIR, batch_size=32, training=False)
    
    print("====== 步骤 2: 初始化 3 通道输入 LeNet 模型 ======")
    garbage_net = LeNet5ForGarbage(num_class=26)
    
    print("====== 步骤 3: 加载预训练模型并【同时过滤 conv1 和 fc3】 ======")
    pretrained_ckpt = "./checkpoint/lenet_mnist-3_1875.ckpt"
    
    if os.path.exists(pretrained_ckpt):
        param_dict = ms.load_checkpoint(pretrained_ckpt)
        filtered_param_dict = {}
        
        for k, v in param_dict.items():
            # ⭐ 关键微调策略：因为 conv1 形状变了，fc3 输出类数变了，这两个头尾结构不加载，只加载中间骨干
            if "fc3" not in k and "conv1" not in k:
                filtered_param_dict[k] = v
                
        param_not_load, _ = ms.load_param_into_net(garbage_net, filtered_param_dict)
        print(f"成功恢复中间特征提取骨干。未加载（从头训练）的参数: {param_not_load}")
    else:
        print("未找到预训练模型，网络将从头训练。")
        
    print("====== 步骤 4: 定义微调所需的损失函数与优化器 ======")
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Momentum(garbage_net.trainable_params(), learning_rate=0.005, momentum=0.9)
    
    model = Model(garbage_net, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})
    
    print("====== 步骤 5: 执行微调训练 ======")
    model.train(epoch=15, 
                train_dataset=ds_train, 
                callbacks=[LossMonitor(per_print_times=10), TimeMonitor(data_size=ds_train.get_dataset_size())], 
                dataset_sink_mode=False)
    print("垃圾分类任务微调成功完成！\n")

    print("====== 步骤 6: 开始在测试集上评估模型 ======")
    metrics = model.eval(ds_test, dataset_sink_mode=False)
    
    print("\n" + "="*40)
    print(f"测试集最终分类准确率 (Accuracy): {metrics['accuracy']:.4%}")
    print("="*40)

if __name__ == "__main__":
    run_finetune_and_test()