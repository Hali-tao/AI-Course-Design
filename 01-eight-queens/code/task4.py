import time
from collections import deque

# ==========================================
# 1. 经典 DFS（深度优先搜索 / 回溯法）实现
# ==========================================
def solve_dfs():
    solutions = []
    
    def is_safe(board, row, col):
        # 检查前面的每一行，看是否与当前位置 (row, col) 冲突
        for i in range(row):
            # board[i] == col 检查列冲突
            # abs(board[i] - col) == abs(i - row) 检查对角线冲突
            if board[i] == col or abs(board[i] - col) == abs(i - row):
                return False
        return True

    def dfs(row, current_board):
        if row == 8:
            solutions.append(tuple(current_board))
            return
        
        for col in range(8):
            if is_safe(current_board, row, col):
                current_board.append(col)  # 放置皇后
                dfs(row + 1, current_board) # 递归下一行
                current_board.pop()        # 回溯（撤销放置）

    start_time = time.time()
    dfs(0, [])
    end_time = time.time()
    
    print(f"【经典 DFS 】求解出的合法解总数: {len(solutions)} 种")
    print(f"【经典 DFS 】耗时: {end_time - start_time:.6f} 秒\n")
    return solutions

# ==========================================
# 2. 经典 BFS（广度优先搜索）实现
# ==========================================
def solve_bfs():
    solutions = []
    # 队列里初始存放一个空棋盘状态 []
    queue = deque([[]])
    
    def is_safe(board, row, col):
        for i in range(row):
            if board[i] == col or abs(board[i] - col) == abs(i - row):
                return False
        return True

    start_time = time.time()
    
    while queue:
        current_board = queue.popleft()
        current_row = len(current_board)
        
        # 如果已经成功放置了 8 个皇后，说明找到了一个合法解
        if current_row == 8:
            solutions.append(tuple(current_board))
            continue
        
        # 尝试在当前行（current_row）的每一列放置皇后
        for col in range(8):
            if is_safe(current_board, current_row, col):
                # 产生一个新状态并推入队列
                queue.append(current_board + [col])
                
    end_time = time.time()
    
    print(f"【经典 BFS 】求解出的合法解总数: {len(solutions)} 种")
    print(f"【经典 BFS 】耗时: {end_time - start_time:.6f} 秒\n")
    return solutions

if __name__ == "__main__":
    dfs_sols = solve_dfs()
    bfs_sols = solve_bfs()