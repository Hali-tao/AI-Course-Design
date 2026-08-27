import os
import mindspore as ms
from mindspore import nn, context, dataset
from mindspore.train import Model
from mindspore.train.callback import LossMonitor, TimeMonitor
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV

# =====================================================================
# 步骤 0: 环境与计算硬件配置
# =====================================================================
context.set_context(mode=context.GRAPH_MODE)
ms.set_device("CPU")  # 如果安装了GPU，可以改为 "GPU"

# 数据集路径配置
GARBAGE_TRAIN_DIR = "./data_en/train/"   # 训练集路径
GARBAGE_TEST_DIR = "./data_en/test/"     # ⭐ 新增：测试集路径

# =====================================================================
# 步骤 1: 加载与优化 26 类垃圾分类数据集
# =====================================================================
def create_garbage_dataset(data_path, batch_size=32, training=True):
    """
    针对 26 类垃圾分类任务的数据集预处理流水线
    """
    # 采用 ImageFolderDataset 自动读取按文件夹分类的垃圾图片
    ds = dataset.ImageFolderDataset(data_path, num_parallel_workers=4, shuffle=training)
    
    # 图像预处理增强管道
    transform = [
        CV.Decode(),                           # 1. 解码图片
        CV.ToPIL(),                            # 2. 转换为 PIL Image 格式（解决冲突的关键步骤）
        CV.Resize((32, 32)),                   # 3. 缩放到 32x32
        CV.Grayscale(num_output_channels=1),   # 4. 转换为单通道灰度图
        CV.ToTensor(),                         # 5. 转换为张量并自动缩放到 [0.0, 1.0]
        CV.Normalize(mean=(0.5,), std=(0.5,))  # 6. 标准化处理
    ]
    
    type_cast_op = C.TypeCast(ms.int32)        # 将分类标签转换为 int32
    
    ds = ds.map(operations=transform, input_columns="image")
    ds = ds.map(operations=type_cast_op, input_columns="label")
    
    # 分批次打包（对于测试集，通常不丢弃余数，但为保持代码一致这里先设为 True 或 False 均可）
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds

# =====================================================================
# 步骤 2: 初始化垃圾分类网络模型 (26 分类输出)
# =====================================================================
class LeNet5ForGarbage(nn.Cell):
    def __init__(self, num_class=26):
        super(LeNet5ForGarbage, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, pad_mode='valid')
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
    print("====== 步骤 1: 加载 26 类垃圾分类 数据集 ======")
    ds_train = create_garbage_dataset(GARBAGE_TRAIN_DIR, batch_size=32, training=True)
    ds_test = create_garbage_dataset(GARBAGE_TEST_DIR, batch_size=32, training=False) # ⭐ 新增：加载测试集
    
    print(f"训练集批次数量: {ds_train.get_dataset_size()}")
    print(f"测试集批次数量: {ds_test.get_dataset_size()}")
    
    print("====== 步骤 2: 初始化垃圾分类网络模型 ======")
    garbage_net = LeNet5ForGarbage(num_class=26)
    
    print("====== 步骤 3: 加载预训练模型并过滤分类头 ======")
    pretrained_ckpt = "./checkpoint/lenet_mnist-3_1875.ckpt"
    
    if os.path.exists(pretrained_ckpt):
        param_dict = ms.load_checkpoint(pretrained_ckpt)
        filtered_param_dict = {}
        for k, v in param_dict.items():
            if "fc3" not in k:
                filtered_param_dict[k] = v
        param_not_load, _ = ms.load_param_into_net(garbage_net, filtered_param_dict)
        print(f"成功加载预训练骨干网络。未加载参数: {param_not_load}")
    else:
        print("未找到预训练模型，网络将从头训练。")
        
    print("====== 步骤 4: 定义微调所需的损失函数与优化器 ======")
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Momentum(garbage_net.trainable_params(), learning_rate=0.002, momentum=0.9)
    
    # 在 metrics 中指定 'accuracy'，这样 Model 内部就知道如何计算准确率
    model = Model(garbage_net, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})
    
    print("====== 步骤 5: 执行微调训练 ======")
    model.train(epoch=5, 
                train_dataset=ds_train, 
                callbacks=[LossMonitor(per_print_times=10), TimeMonitor(data_size=ds_train.get_dataset_size())], 
                dataset_sink_mode=False)
    print("垃圾分类任务微调成功完成！\n")

    # =====================================================================
    # ⭐ 新增步骤 6: 在独立测试集上评估模型结果
    # =====================================================================
    print("====== 步骤 6: 开始在测试集上评估模型 ======")
    metrics = model.eval(ds_test, dataset_sink_mode=False)
    
    print("\n" + "="*40)
    print(f"测试集分类准确率 (Accuracy): {metrics['accuracy']:.4%}")
    print("="*40)

if __name__ == "__main__":
    run_finetune_and_test()