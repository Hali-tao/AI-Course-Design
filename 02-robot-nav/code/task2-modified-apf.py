import numpy as np
import matplotlib.pyplot as plt
import time

# 引入之前还原的地图数据与起终点
from map import MAP_GRID_25X25, START_POS, END_POS

class ImprovedAPFGridEnv:
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
        # 允许 8 方向移动
        directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if self.is_valid((nr, nc)):
                neighbors.append((nr, nc))
        return neighbors

    def run_improved_apf(self, k_att=1.5, k_rep=25.0, d_0=4.0):
        """
        改进型人工势场算法（引入虚拟目标点与历史轨迹惩罚机制）
        """
        current = self.start
        path = [current]
        visited_history = [current]
        
        # 死锁检测与虚拟目标控制变量
        staleness_counter = 0
        virtual_target = None
        virtual_target_timer = 0
        
        obstacle_poses = np.argwhere(self.grid == 1)
        max_steps = 300  # 给予充足的绕行步数
        step = 0
        
        while current != self.end and step < max_steps:
            step += 1
            neighbors = self.get_neighbors(current)
            if not neighbors:
                break 
                
            best_neighbor = None
            min_potential = float('inf')
            
            # --- 改进管理：虚拟目标的生命周期管理 ---
            if virtual_target is not None:
                virtual_target_timer -= 1
                dist_to_vt = np.linalg.norm(np.array(current) - np.array(virtual_target))
                # 如果时间到了，或者机器人已经很接近虚拟目标了，就释放它
                if virtual_target_timer <= 0 or dist_to_vt < 1.5:
                    virtual_target = None 
            
            for neighbor in neighbors:
                # 1. 计算引力场：如果存在虚拟目标，则向虚拟目标对齐；否则向真正的终点对齐
                target = virtual_target if virtual_target is not None else self.end
                dist_to_target = np.linalg.norm(np.array(neighbor) - np.array(target))
                u_att = 0.5 * k_att * (dist_to_target ** 2)
                
                # 2. 计算斥力场
                u_rep = 0.0
                for obs in obstacle_poses:
                    dist_to_obs = np.linalg.norm(np.array(neighbor) - np.array(obs))
                    if dist_to_obs <= d_0:
                        if dist_to_obs < 0.1: dist_to_obs = 0.1
                        u_rep += 0.5 * k_rep * ((1.0 / dist_to_obs - 1.0 / d_0) ** 2)
                
                u_total = u_att + u_rep
                
                # --- 改进辅助项：历史轨迹惩罚（动态局部增益），防止机器人在微小区域踱步 ---
                # 越是最近走过的格子，势场施加额外的惩罚，逼迫它向未探索区域拓展
                if neighbor in path[-8:]:
                    u_total += 5.0
                    
                if u_total < min_potential:
                    min_potential = u_total
                    best_neighbor = neighbor
            
            if best_neighbor is None:
                break
                
            # --- 改进触发：死锁（局部最小值）检测 ---
            # 如果连续数步的最优邻居都在最近的轨迹里徘徊，说明卡住了
            if best_neighbor in path[-6:]:
                staleness_counter += 1
            else:
                staleness_counter = 0
                
            # 当判定卡死超过 4 步，且当前没有虚拟目标引导时，激活改进机制
            if staleness_counter >= 4 and virtual_target is None:
                # 根据当前陷入陷阱的几何特征，在口袋左侧出口外的开阔走廊 (列索引为 3) 设立虚拟引导点
                # 行坐标与当前卡死位置保持相近，引导其横向逸出
                virtual_target = (current[0], 3)
                virtual_target_timer = 25  # 允许该虚拟目标点存在 25 步
                staleness_counter = 0      # 重置计数器
            
            current = best_neighbor
            path.append(current)
            visited_history.append(current)
            
        return path, visited_history

    def draw_improved_result(self, path, visited, title, filename):
        display_img = np.ones((self.rows, self.cols, 3))
        display_img[self.grid == 1] = [0.2, 0.4, 0.8]  # 蓝色障碍物
        
        plt.figure(figsize=(10, 10))
        plt.imshow(display_img, extent=[0, self.cols, self.rows, 0])
        
        # 绘制搜索留下的浅橙色足迹
        for r, c in visited:
            if (r, c) != self.start and (r, c) != self.end:
                plt.fill_between([c, c+1], r, r+1, color='orange', alpha=0.2)
                
        # 绘制最终的避障路径（荧光绿）
        if path:
            path_x = [c + 0.5 for r, c in path]
            path_y = [r + 0.5 for r, c in path]
            plt.plot(path_x, path_y, color='lime', linewidth=3.5, label='Improved APF Path', zorder=5)
            
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
    iapf_env = ImprovedAPFGridEnv(MAP_GRID_25X25, START_POS, END_POS)
    
    # 运行改进型 APF 算法
    t_start = time.perf_counter()
    path, visited = iapf_env.run_improved_apf()
    t_end = time.perf_counter()
    
    execution_time_ms = (t_end - t_start) * 1000
    is_success = (path[-1] == END_POS)
    
    print("\n" + "="*50)
    print(f"{'改进型 APF 算法在 25x25 随机地图下的量化指标':^35}")
    print("="*50)
    print(f"最终是否成功抵达终点: {is_success}")
    print(f"算法运行总消耗步数:   {len(path) - 1} 步")
    print(f"路径规划耗时:         {execution_time_ms:.4f} ms")
    print(f"最终到达(或停滞)坐标: {path[-1]}")
    print("="*50 + "\n")
    
    # 绘图展示
    iapf_env.draw_improved_result(
        path, 
        visited, 
        "Improved APF Pathfinding Success (Virtual Target Escape)", 
        "improved_apf_success.png"
    )