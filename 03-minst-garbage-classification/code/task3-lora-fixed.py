import os
import time
import psutil
import mindspore as ms
from mindspore import nn, context, dataset
from mindspore.train import Model
from mindspore.train.callback import LossMonitor, Callback
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV

# 环境配置
context.set_context(mode=context.GRAPH_MODE)
ms.set_device("CPU")

# 路径配置
GARBAGE_TRAIN_DIR = "./data_en/train/"
GARBAGE_TEST_DIR = "./data_en/test/"
PRETRAINED_CKPT = "./checkpoint/lenet_mnist-3_1875.ckpt"

class MemoryTrackerCallback(Callback):
    def __init__(self):
        super(MemoryTrackerCallback, self).__init__()
        self.process = psutil.Process(os.getpid())
        self.max_memory = 0.0

    def epoch_end(self, run_context):
        current_mem = self.process.memory_info().rss / (1024.0 * 1024.0)
        if current_mem > self.max_memory:
            self.max_memory = current_mem

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

class LoRADense(nn.Cell):
    def __init__(self, in_channels, out_channels, r=4, alpha=8):
        super(LoRADense, self).__init__()
        self.original_dense = nn.Dense(in_channels, out_channels)
        self.original_dense.weight.requires_grad = False 
        self.original_dense.bias.requires_grad = True 
        
        self.lora_A = ms.Parameter(ms.common.initializer.initializer('normal', [in_channels, r]), name="lora_A")
        self.lora_B = ms.Parameter(ms.common.initializer.initializer('zeros', [r, out_channels]), name="lora_B")
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

def load_and_filter_weights(net):
    if os.path.exists(PRETRAINED_CKPT):
        param_dict = ms.load_checkpoint(PRETRAINED_CKPT)
        net_param_shapes = {name: param.shape for name, param in net.parameters_and_names()}
        filtered_dict = {}
        print("\n" + "-"*15 + " 预训练权重热加载日志 (LoRA微调) " + "-"*15)
        for k, v in param_dict.items():
            if any(x in k for x in ["moments", "global_step", "learning_rate", "momentum"]):
                continue
            
            target_key = k
            if "fc1" in k and "original_dense" not in k:
                target_key = k.replace("fc1", "fc1.original_dense")
            elif "fc2" in k and "original_dense" not in k:
                target_key = k.replace("fc2", "fc2.original_dense")
            
            if "conv1.weight" in k and net_param_shapes.get(target_key) == (6, 3, 5, 5):
                v_expanded = ms.ops.tile(v, (1, 3, 1, 1)) / 3.0
                filtered_dict[target_key] = ms.Parameter(v_expanded, name=target_key)
                print(f"【成功适配】{k} (6,1,5,5) -> 扩展均分适配为 3 通道 (6,3,5,5)")
                continue
            
            if target_key in net_param_shapes and v.shape == net_param_shapes[target_key]:
                filtered_dict[target_key] = v
                print(f"【成功加载】{k} -> {target_key}")
        ms.load_param_into_net(net, filtered_dict)
        print("-" * 62 + "\n")

if __name__ == "__main__":
    print("\n▶▶▶ 正在启动策略: 【LORA 微调】 ◀◀◀")
    ds_train = create_garbage_dataset(GARBAGE_TRAIN_DIR, batch_size=32, training=True)
    ds_test = create_garbage_dataset(GARBAGE_TEST_DIR, batch_size=32, training=False)
    
    net = LoRALeNet5(num_class=26)
    load_and_filter_weights(net)
    
    trainable_params = filter(lambda p: p.requires_grad, net.get_parameters())
    num_trainable = sum([p.size for p in net.get_parameters() if p.requires_grad])
    print(f"当前模式下实际可训练参数量: {num_trainable} 个")
    
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Momentum(trainable_params, learning_rate=0.002, momentum=0.9)
    model = Model(net, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})
    
    loss_cb = LossMonitor(per_print_times=ds_train.get_dataset_size())
    mem_cb = MemoryTrackerCallback()
    
    start_time = time.time()
    model.train(epoch=25, train_dataset=ds_train, callbacks=[loss_cb, mem_cb], dataset_sink_mode=False)
    end_time = time.time()
    
    metrics = model.eval(ds_test, dataset_sink_mode=False)
    
    print("\n" + "="*20 + " 实验指标单项看板 (LoRA微调) " + "="*20)
    print(f"可训练参数量: {num_trainable}")
    print(f"总训练耗时  : {end_time - start_time:.2f} 秒")
    print(f"CPU内存峰值 : {mem_cb.max_memory:.2f} MB")
    print(f"测试集最终准确率: {metrics['accuracy']:.4%}")
    print("=" * 68)