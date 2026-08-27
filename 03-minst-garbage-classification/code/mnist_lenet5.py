import os
import mindspore as ms
from mindspore import nn, context, dataset
from mindspore.train import Model
from mindspore.train.callback import LossMonitor, CheckpointConfig, ModelCheckpoint
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV

# 设置运行环境：CPU或GPU
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")  # 如果装了GPU就改成"GPU"


# from download import download

# # 下载MNIST数据集（MindSpore官方镜像，约10MB）
# url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip"
# path = download(url, "./", kind="zip", replace=True)
# print(f"数据集已下载并解压到: {path}")

# 数据集解压后的路径
DATA_DIR = "./MNIST_Data/"


def create_dataset(data_path, batch_size=32, training=True):
    """
    创建MNIST数据集
    - data_path: 数据存放路径
    - batch_size: 批次大小
    - training: 是否训练模式（训练集需要打乱顺序）
    """
    if training:
        ds = dataset.MnistDataset(data_path + "train")
        ds = ds.shuffle(buffer_size=64)  # 打乱顺序，避免过拟合
    else:
        ds = dataset.MnistDataset(data_path + "test")
    
    # 数据预处理操作
    # 1. 缩放到32x32（LeNet5的原始输入要求）
    # 2. 归一化：公式是 (x - mean) / std，这里用的mean=0.1307, std=0.3081是MNIST的标准值
    # 3. 转换维度顺序：HWC → CHW（MindSpore要求）
    transform = [
        CV.Resize((32, 32)),
        CV.Rescale(1.0 / 255.0, 0),  # 将像素值从[0,255]缩放到[0,1]
        CV.Normalize(mean=(0.1307,), std=(0.3081,)),  # 标准化
        CV.HWC2CHW()
    ]
    
    # 标签转换：转成int32类型
    type_cast_op = C.TypeCast(ms.int32)
    
    # 应用预处理
    ds = ds.map(operations=transform, input_columns="image")
    ds = ds.map(operations=type_cast_op, input_columns="label")
    
    # 分批：每batch_size张图片作为一个批次
    ds = ds.batch(batch_size, drop_remainder=True)
    
    return ds

class LeNet5(nn.Cell):
    """
    LeNet-5 网络结构
    卷积层 → 池化层 → 卷积层 → 池化层 → 全连接层×3
    """
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        
        # 第一层卷积：输入1通道（灰度图），输出6通道，卷积核5×5
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        # 第二层卷积：输入6通道，输出16通道，卷积核5×5
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        
        # 全连接层（也叫Dense层）
        # 16*5*5 是第二层卷积+池化后的特征图大小
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, num_class)  # 输出10类（数字0-9）
        
        # 激活函数和池化层
        self.relu = nn.ReLU()           # 引入非线性，让网络能学习复杂特征
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样，降维
        self.flatten = nn.Flatten()     # 把多维特征展平成一维，接全连接层
    
    def construct(self, x):
        """前向传播：定义数据如何流过网络"""
        # 第一组：卷积 → ReLU → 池化
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        
        # 第二组：卷积 → ReLU → 池化
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        
        # 展平
        x = self.flatten(x)
        
        # 全连接层：120 → 84 → 10
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)  # 最终输出10个数字的分数
        
        return x

# 验证网络结构是否正确
net = LeNet5()
print(net)

def train():
    """训练LeNet5模型"""
    # 1. 创建数据集
    print("加载训练集...")
    ds_train = create_dataset(DATA_DIR, batch_size=32, training=True)
    ds_eval = create_dataset(DATA_DIR, batch_size=32, training=False)
    
    print(f"训练集批次数量: {ds_train.get_dataset_size()}")
    print(f"测试集批次数量: {ds_eval.get_dataset_size()}")
    
    # 2. 定义网络
    network = LeNet5(num_class=10)
    
    # 3. 定义损失函数（交叉熵损失）
    # sparse=True表示标签是整数而非one-hot编码
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    
    # 4. 定义优化器（带动量的SGD）
    # learning_rate: 学习率，控制每次更新的步长
    # momentum: 动量，帮助加速收敛并跳出局部极小值
    optimizer = nn.Momentum(network.trainable_params(), learning_rate=0.01, momentum=0.9)
    
    # 5. 封装成Model
    model = Model(network, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})
    
    # 6. 配置模型保存（每训练完1个epoch保存一次）
    config_ck = CheckpointConfig(save_checkpoint_steps=ds_train.get_dataset_size(), 
                                  keep_checkpoint_max=5)
    ckpoint_cb = ModelCheckpoint(prefix="lenet_mnist", directory="./checkpoint/", config=config_ck)
    
    # 7. 开始训练
    print("开始训练...")
    model.train(epoch=3,              # 训练3轮
                train_dataset=ds_train,
                callbacks=[LossMonitor(per_print_times=ds_train.get_dataset_size()), ckpoint_cb],
                dataset_sink_mode=False)  # CPU模式下设为False
    
    # 8. 评估模型
    print("评估模型...")
    metrics = model.eval(ds_eval, dataset_sink_mode=False)
    print(f"测试集准确率: {metrics['accuracy']:.4%}")
    
    print("训练完成！模型已保存到 ./checkpoint/lenet_mnist-3_*.ckpt")
    return model

# 运行训练
if __name__ == "__main__":
    train()