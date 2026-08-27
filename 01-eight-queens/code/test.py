import sys
print(f"Python 路径: {sys.executable}")

print(f"Python 版本: {sys.version}")

print("\n正在测试 kanren 导入...")

try:
    from kanren import run, eq, var, membero
    print("✓ kanren 导入成功")
    
    # 简单测试
    x = var()
    result = run(0, x, eq(x, 5))
    print(f"\n逻辑推理测试: 找出 x 使得 x == 5")
    print(f"结果: {result}")
    
    if result == (5,):
        print("✓ 逻辑推理正确")
    else:
        print("✗ 逻辑推理异常")
        
except ImportError as e:
    print(f"✗ kanren 导入失败: {e}")
    print("\n请先安装 kanren:")
    print("  pip install kanren")
    
print("\n" + "="*40)
print("环境测试完成")