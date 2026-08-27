import numpy as np
import matplotlib.pyplot as plt
import heapq
import time

# 从自定义地图模块引入 25x25 矩阵及起终点
from map import MAP_GRID_25X25, START_POS, END_POS

class DijkstraGridEnv:
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

    def run_dijkstra(self):
        """Dijkstra 算法核心实现"""
        count = 0
        # 优先队列元素格式: (累计代价g_score, 唯一计数器, 当前节点坐标)
        open_set = [(0, count, self.start)]
        came_from = {self.start: None}
        g_score = {self.start: 0}
        
        expansion_order = {}
        order_counter = 1
        
        while open_set:
            current_g, _, current = heapq.heappop(open_set)
            
            if current in expansion_order:
                continue
                
            expansion_order[current] = order_counter
            order_counter += 1
            
            if current == self.end:
                break
                
            for neighbor in self.get_neighbors(current):
                # 4方向移动，每走一步代价恒定为 1
                tentative_g = current_g + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    count += 1
                    heapq.heappush(open_set, (tentative_g, count, neighbor))
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
        """绘制符合 25x25 地图学术规范的 Dijkstra 扩散热力图"""
        heatmap_matrix = np.full((self.rows, self.cols), np.nan)
        for (r, c), order in expansion_order.items():
            heatmap_matrix[r, c] = order
            
        display_img = np.ones((self.rows, self.cols, 3))
        display_img[self.grid == 1] = [0.2, 0.4, 0.8]  # 蓝色障碍物
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(display_img, extent=[0, self.cols, self.rows, 0])
        
        vmax = max(expansion_order.values()) if expansion_order else 1
        masked_heatmap = np.ma.masked_where(np.isnan(heatmap_matrix), heatmap_matrix)
        
        im = ax.imshow(masked_heatmap, cmap='YlOrRd_r', extent=[0, self.cols, self.rows, 0], vmin=1, vmax=vmax, alpha=0.8)
        
        if path:
            path_x = [c + 0.5 for r, c in path]
            path_y = [r + 0.5 for r, c in path]
            ax.plot(path_x, path_y, color='lime', linewidth=4, label='Dijkstra Shortest Path', zorder=5)
            
        ax.text(self.start[1]+0.5, self.start[0]+0.5, 'S', va='center', ha='center', color='white', weight='bold', fontsize=14, bbox=dict(facecolor='red', edgecolor='none'))
        ax.text(self.end[1]+0.5, self.end[0]+0.5, 'E', va='center', ha='center', color='black', weight='bold', fontsize=14, bbox=dict(facecolor='lightgray', edgecolor='none'))
        
        ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xticks(range(self.cols+1))
        ax.set_yticks(range(self.rows+1))
        ax.set_title("Dijkstra Node Expansion Wavefront Heatmap (25x25 Grid)", fontsize=14, pad=15)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Expansion Sequence (Steps)', rotation=270, labelpad=15, fontsize=11)
        
        plt.legend(loc='upper right', fontsize=11)
        plt.tight_layout()
        plt.savefig(save_name, dpi=300)
        print(f"📷 Dijkstra 结果已成功渲染并保存至: {save_name}")
        plt.show()

if __name__ == "__main__":
    env = DijkstraGridEnv(MAP_GRID_25X25, START_POS, END_POS)
    
    print("🚀 开始执行 Dijkstra 算法...")
    t_start = time.perf_counter()
    dijk_order, dijk_path = env.run_dijkstra()
    t_end = time.perf_counter()
    
    execution_time_ms = (t_end - t_start) * 1000
    path_len = len(dijk_path) - 1 if dijk_path else 0
    visited_nodes = len(dijk_order)
    
    print("\n" + "="*50)
    print(f"{'Dijkstra 算法 25x25 地图量化报告':^40}")
    print("="*50)
    print(f"最终是否成功抵达终点: {len(dijk_path) > 0}")
    print(f"规划最短路径步数:     {path_len} 步")
    print(f"算法总扩展节点数:     {visited_nodes} 个")
    print(f"算法纯求解耗时:       {execution_time_ms:.4f} ms")
    print("="*50 + "\n")
    
    env.draw_heatmap(dijk_order, dijk_path, "dijkstra_25x25_heatmap.png")