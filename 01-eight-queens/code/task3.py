import time
from kanren import run, var, membero

# ==========================================
# 1. 自主实现 neq 函数
# ==========================================
def neq(x, y):
    """自定义逻辑不等于目标"""
    def neq_goal(sub):
        val_x = sub.get(x, x)
        val_y = sub.get(y, y)
        if not isinstance(val_x, type(x)) and not isinstance(val_y, type(y)):
            if val_x == val_y:
                return
        yield sub
    return neq_goal

# ==========================================
# 2. 自主实现对角线不相等约束
# ==========================================
def neq_diagonal(q_i, q_j, i, j):
    """自定义对角线不相等目标: |q_i - q_j| != |i - j|"""
    def diagonal_goal(sub):
        val_i = sub.get(q_i, q_i)
        val_j = sub.get(q_j, q_j)
        if not isinstance(val_i, type(q_i)) and not isinstance(val_j, type(q_j)):
            if abs(val_i - val_j) == abs(i - j):
                return
        yield sub
    return diagonal_goal

# ==========================================
# 3. 结合自定义函数求解八皇后
# ==========================================
def solve_queens_with_custom_neq():
    queens = [var(f'q_{i}') for i in range(8)]
    
    # 动态构建交错的约束流（赋值一个，立刻检查与之相关的列和对角线）
    all_goals = []
    for i in range(8):
        all_goals.append(membero(queens[i], range(8)))
        for j in range(i):
            # 使用自主实现的 neq 函数约束列号 
            all_goals.append(neq(queens[j], queens[i]))
            # 使用自主实现的对角线函数约束对角线
            all_goals.append(neq_diagonal(queens[j], queens[i], j, i))

    print("正在启动包含自定义 neq 函数的逻辑编程求解...")
    start_time = time.time()
    
    solutions = run(0, queens, *all_goals)
    
    end_time = time.time()
    
    print(f"【自定义neq逻辑编程】求解出的合法解总数: {len(solutions)} 种")
    print(f"【自定义neq逻辑编程】耗时: {end_time - start_time:.6f} 秒\n")

if __name__ == "__main__":
    solve_queens_with_custom_neq()