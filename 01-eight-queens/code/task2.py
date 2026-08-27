import time
from kanren import run, var, membero

def solve_queens_pure_logic():
    # 1. 声明 8 个逻辑变量，分别代表 0~7 行皇后所在的列
    queens = [var(f'q_{i}') for i in range(8)]
    
    # 2. 约束：每个皇后的列坐标必须在 0~7 之间
    # membero(x, coll) 表示 x 必须是迭代器 coll 中的一个元素
    domain_goals = [membero(q, range(8)) for q in queens]
    
    # 3. 约束：任意两个皇后不能在同一列，也不能在同一对角线
    constraint_goals = []
    
    # 构建一个自定义的逻辑目标，用于动态检查两个已经绑定值的皇后是否冲突
    def safe_pair_goal(q_i, q_j, i, j):
        def goal(sub):
            # 获取当前状态下，两个逻辑变量的实例化值
            val_i = sub.get(q_i, q_i)
            val_j = sub.get(q_j, q_j)
            
            # 如果两个变量都已经被赋予了具体的数字，则进行冲突检查
            if isinstance(val_i, int) and isinstance(val_j, int):
                # 检查列冲突
                if val_i == val_j:
                    return
                # 检查对角线冲突
                if abs(val_i - val_j) == abs(i - j):
                    return
            # 如果没冲突，或者变量还没完全绑定，则保留当前状态继续向下搜索
            yield sub
        return goal

    # 双重循环，为任意两行（i, j）之间建立逻辑约束目标
    for i in range(8):
        for j in range(i + 1, 8):
            constraint_goals.append(safe_pair_goal(queens[i], queens[j], i, j))
            
    # 4. 将所有的值域约束和冲突约束合并为一个庞大的“逻辑约束网”
    # 注意：在 kanren 中，约束的顺序至关重要。
    # 我们交叉放置 membero 和 safe_pair_goal，让 kanren 赋值一个就立刻检查一个（即边生成边剪枝）
    all_goals = []
    for i in range(8):
        all_goals.append(domain_goals[i])
        # 每多确定一个皇后的位置，就和之前所有已确定的皇后进行冲突对比
        for j in range(i):
            all_goals.append(safe_pair_goal(queens[j], queens[i], j, i))

    print("正在启动纯逻辑编程求解（动态约束传递与回溯）...")
    start_time = time.time()
    
    # 5. 运行 kanren 求解引擎
    # *all_goals 将整个约束网络解包传入，queens 元组作为我们要获取的答案形式
    solutions = run(0, queens, *all_goals)
    
    end_time = time.time()
    
    print(f"【纯逻辑编程】求解出的合法解总数: {len(solutions)} 种")
    print(f"【纯逻辑编程】耗时: {end_time - start_time:.6f} 秒\n")

if __name__ == "__main__":
    solve_queens_pure_logic()