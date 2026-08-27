import numpy as np
import matplotlib.pyplot as plt
import time

# 引入上一阶段还原的地图数据与起终点
from map import MAP_GRID_25X25, START_POS, END_POS

class TraditionalAPFGridEnv:
    def __init__(self, grid_matrix, start, end):
        self.grid = np.array(grid_matrix)
        self.rows, self.cols = self.grid.shape
        self.start = start
        self.end = end
        
    def is_valid(self, pos):
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r, c] == 0

    def get_neighbors(self, pos):
        r, c = pos
        # 允许 8 方向移动（上下左右 + 对角线），使势场运动轨迹更平滑
        directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if self.is_valid((nr, nc)):
                neighbors.append((nr, nc))
        return neighbors

    def run_traditional_apf(self, k_att=1.0, k_rep=15.0, d_0=4.0):
        """
        传统人工势场算法
        :param k_att: 引力增益系数
        :param k_rep: 斥力增益系数
        :param d_0: 障碍物影响的距离阈值
        """
        current = self.start
        path = [current]
        visited_history = [current]
        
        # 获取地图中所有障碍物的坐标，用于计算斥力
        obstacle_poses = np.argwhere(self.grid == 1)
        
        max_steps = 200  # 设置最大步数限制，防止死锁时产生无限循环
        step = 0
        
        while current != self.end and step < max_steps:
            step += 1
            neighbors = self.get_neighbors(current)
            if not neighbors:
                break # 无路可走
                
            best_neighbor = None
            min_potential = float('inf')
            
            # 计算每个邻居节点的势场值
            for neighbor in neighbors:
                # 1. 计算引力场 (经典二次型引力公式)
                dist_to_target = np.linalg.norm(np.array(neighbor) - np.array(self.end))
                u_att = 0.5 * k_att * (dist_to_target ** 2)
                
                # 2. 计算斥力场 (经典斥力公式)
                u_rep = 0.0
                for obs in obstacle_poses:
                    dist_to_obs = np.linalg.norm(np.array(neighbor) - np.array(obs))
                    if dist_to_obs <= d_0:
                        if dist_to_obs < 0.1: dist_to_obs = 0.1  # 防止除以0
                        u_rep += 0.5 * k_rep * ((1.0 / dist_to_obs - 1.0 / d_0) ** 2)
                
                # 3. 总势场值
                u_total = u_att + u_rep
                
                # 选择总势场最小（即能量最低）的邻居节点
                if u_total < min_potential:
                    min_potential = u_total
                    best_neighbor = neighbor
            
            if best_neighbor is None:
                break
                
            current = best_neighbor
            path.append(current)
            visited_history.append(current)
            
        return path, visited_history

    def draw_apf_result(self, path, visited, title, filename):
        display_img = np.ones((self.rows, self.cols, 3))
        display_img[self.grid == 1] = [0.2, 0.4, 0.8]  # 蓝色障碍物
        
        plt.figure(figsize=(9, 9))
        plt.imshow(display_img, extent=[0, self.cols, self.rows, 0])
        
        # 绘制搜索留下的浅橙色足迹
        for r, c in visited:
            if (r, c) != self.start and (r, c) != self.end:
                plt.fill_between([c, c+1], r, r+1, color='orange', alpha=0.3)
                
        # 绘制最终的避障路径（荧光绿）
        if path:
            path_x = [c + 0.5 for r, c in path]
            path_y = [r + 0.5 for r, c in path]
            plt.plot(path_x, path_y, color='lime', linewidth=3.5, label='APF Path', zorder=5)
            
        plt.text(self.start[1]+0.5, self.start[0]+0.5, 'S', va='center', ha='center', color='white', weight='bold', fontsize=14, bbox=dict(facecolor='red', edgecolor='none'))
        plt.text(self.end[1]+0.5, self.end[0]+0.5, 'E', va='center', ha='center', color='black', weight='bold', fontsize=14, bbox=dict(facecolor='lightgray', edgecolor='none'))
        
        plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
        plt.xticks(range(self.cols+1))
        plt.yticks(range(self.rows+1))
        plt.title(title, fontsize=12, pad=15)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()

# --- 主测试程序 ---
if __name__ == "__main__":
    # 初始化环境
    apf_env = TraditionalAPFGridEnv(MAP_GRID_25X25, START_POS, END_POS)
    
    # 运行传统 APF 算法
    t_start = time.perf_counter()
    path, visited = apf_env.run_traditional_apf()
    t_end = time.perf_counter()
    
    execution_time_ms = (t_end - t_start) * 1000
    is_success = (path[-1] == END_POS)
    
    print("\n" + "="*50)
    print(f"{'传统 APF 算法在 25x25 随机地图下的量化指标':^35}")
    print("="*50)
    print(f"最终是否成功抵达终点: {is_success}")
    print(f"算法运行总消耗步数:   {len(path) - 1} 步")
    print(f"路径规划耗时:         {execution_time_ms:.4f} ms")
    print(f"最终停滞(死锁)坐标:   {path[-1]}")
    print("="*50 + "\n")
    
    # 绘图展示
    apf_env.draw_apf_result(
        path, 
        visited, 
        "Traditional APF Pathfinding Failure (Local Minimum Lock)", 
        "traditional_apf_failure.png"
    )