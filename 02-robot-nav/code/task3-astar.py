import numpy as np
import matplotlib.pyplot as plt
import heapq
import time

# 从自定义地图模块引入 25x25 矩阵及起终点
from map import MAP_GRID_25X25, START_POS, END_POS

class AStarGridEnv:
    def __init__(self, grid_matrix, start, end):
        self.grid = np.array(grid_matrix)
        self.rows, self.cols = self.grid.shape
        self.start = start
        self.end = end
        
    def get_neighbors(self, pos):
        r, c = pos
        # 严格执行上下左右 4 方向扩展
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr, nc] == 0:
                neighbors.append((nr, nc))
        return neighbors

    def run_astar(self):
        """A* 算法 (严格基于曼哈顿启发距离机制)"""
        def heuristic(p1, p2):
            # 4方向约束下的标准启发式函数
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
            
        count = 0
        open_set = [(0, count, self.start)]
        came_from = {self.start: None}
        g_score = {self.start: 0}
        
        expansion_order = {}
        order_counter = 1
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            if current in expansion_order:
                continue
                
            expansion_order[current] = order_counter
            order_counter += 1
            
            if current == self.end:
                break
                
            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, self.end)
                    count += 1
                    heapq.heappush(open_set, (f_score, count, neighbor))
                    came_from[neighbor] = current
                    
        path = []
        curr = self.end
        if curr in came_from:
            while curr is not None:
                path.append(curr)
                curr = came_from[curr]
            path.reverse()
        return expansion_order, path

    def draw_heatmap(self, expansion_order, path, save_name):
        """绘制符合 25x25 地图学术规范的 A* 启发式收敛热力图"""
        heatmap_matrix = np.full((self.rows, self.cols), np.nan)
        for (r, c), order in expansion_order.items():
            heatmap_matrix[r, c] = order
            
        display_img = np.ones((self.rows, self.cols, 3))
        display_img[self.grid == 1] = [0.2, 0.4, 0.8]  # 蓝色障碍物
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(display_img, extent=[0, self.cols, self.rows, 0])
        
        vmax = max(expansion_order.values()) if expansion_order else 1
        masked_heatmap = np.ma.masked_where(np.isnan(heatmap_matrix), heatmap_matrix)
        
        # 采用明艳的热力图色，可以直观对比出 A* 极窄的搜索扇面
        im = ax.imshow(masked_heatmap, cmap='YlOrRd_r', extent=[0, self.cols, self.rows, 0], vmin=1, vmax=vmax, alpha=0.8)
        
        # 绘制最短路径
        if path:
            path_x = [c + 0.5 for r, c in path]
            path_y = [r + 0.5 for r, c in path]
            ax.plot(path_x, path_y, color='lime', linewidth=4, label='A* Shortest Path', zorder=5)
            
        # 起终点醒目标注
        ax.text(self.start[1]+0.5, self.start[0]+0.5, 'S', va='center', ha='center', color='white', weight='bold', fontsize=14, bbox=dict(facecolor='red', edgecolor='none'))
        ax.text(self.end[1]+0.5, self.end[0]+0.5, 'E', va='center', ha='center', color='black', weight='bold', fontsize=14, bbox=dict(facecolor='lightgray', edgecolor='none'))
        
        ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xticks(range(self.cols+1))
        ax.set_yticks(range(self.rows+1))
        ax.set_title("A* Node Expansion Driven Heatmap (25x25 Grid)", fontsize=14, pad=15)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Expansion Sequence (Steps)', rotation=270, labelpad=15, fontsize=11)
        
        plt.legend(loc='upper right', fontsize=11)
        plt.tight_layout()
        plt.savefig(save_name, dpi=300)
        print(f"📷 结果已成功渲染并保存至: {save_name}")
        plt.show()

if __name__ == "__main__":
    env = AStarGridEnv(MAP_GRID_25X25, START_POS, END_POS)
    
    print("🚀 开始执行启发式搜索 (A*)...")
    t_start = time.perf_counter()
    astar_order, astar_path = env.run_astar()
    t_end = time.perf_counter()
    
    # 指标计算
    execution_time_ms = (t_end - t_start) * 1000
    path_len = len(astar_path) - 1 if astar_path else 0
    visited_nodes = len(astar_order)
    
    print("\n" + "="*50)
    print(f"{'A* 算法 25x25 地图量化报告':^40}")
    print("="*50)
    print(f"最终是否成功抵达终点: {len(astar_path) > 0}")
    print(f"规划最短路径步数:     {path_len} 步")
    print(f"算法总扩展节点数:     {visited_nodes} 个")
    print(f"算法纯求解耗时:       {execution_time_ms:.4f} ms")
    print("="*50 + "\n")
    
    # 绘图输出
    env.draw_heatmap(astar_order, astar_path, "astar_25x25_heatmap.png")