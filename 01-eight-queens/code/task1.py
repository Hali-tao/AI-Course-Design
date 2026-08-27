import time
from itertools import permutations
from kanren import Relation, facts, run, var

# ==========================================
# 1. 非逻辑编程方式：穷举所有可能的排列 (Col Positions)
# ==========================================
# 使用 permutations 天然保证了任意两个皇后不在同一列，也不在同一行
all_permutations = list(permutations(range(8)))
print(f"【非逻辑编程】穷举生成的排列总数: {len(all_permutations)} 种")

# ==========================================
# 2. 逻辑编程方式：筛选出符合对角线规则的解
# ==========================================

# 定义一个检查对角线冲突的辅助函数
def is_valid_diagonal(perm):
    """
    检查一个排列是否满足对角线不冲突。
    若两个皇后坐标为 (i, perm[i]) 和 (j, perm[j]),
    则满足 abs(i - j) != abs(perm[i] - perm[j])
    """
    for i in range(8):
        for j in range(i + 1, 8):
            if abs(i - j) == abs(perm[i] - perm[j]):
                return False  # 发生对角线冲突
    return True

# ==========================================
# 3. 利用 kanren 进行逻辑筛选
# ==========================================

# 声明一个逻辑关系：代表“这是一个生成的排列”
permutation_rel = Relation()

# 把【所有】未经过滤的穷举排列（40320种）作为事实录入数据库
facts(permutation_rel, *[(p,) for p in all_permutations])

# 声明逻辑变量
q_solution = var()

# 通过底层构建一个合法的 kanren 目标（Goal）来替代 condition
def diagonal_goal_func(var_sol):
    def goal(sub):
        # 获取当前逻辑变量被绑定（实例化）后的具体值
        val = sub.get(var_sol, var_sol)
        
        # 如果变量已经绑定了具体的排列，并且通过了对角线检查，则保留该状态
        if isinstance(val, tuple) and is_valid_diagonal(val):
            yield sub
    return goal

print("\n正在启动 kanren 逻辑推理与筛选...")
start_logic = time.time()

# 运行推理：寻找同时满足“是排列事实”且“通过自定义对角线逻辑目标”的解
answers = run(0, q_solution, 
              permutation_rel(q_solution), 
              diagonal_goal_func(q_solution))

end_logic = time.time()

# ==========================================
# 4. 结果输出
# ==========================================
print(f"【逻辑编程】筛选出的合法八皇后解总数: {len(answers)} 种")
print(f"【逻辑编程】通过自定义逻辑目标筛选耗时: {end_logic - start_logic:.6f} 秒")