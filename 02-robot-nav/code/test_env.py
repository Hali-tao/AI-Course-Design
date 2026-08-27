import numpy as np
import pandas as pd
import sklearn
import torch
import matplotlib.pyplot as plt

print("--- 环境检查结果 ---")
print(f"Python 版本:  3.10")
print(f"NumPy 版本:   {np.__version__}")
print(f"Pandas 版本:  {pd.__version__}")
print(f"Sklearn 版本: {sklearn.__version__}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}") # 如果装了GPU版且有显卡，这里应显示 True
print("🎉 所有基础库导入成功，环境配置完成！")