import numpy as np
import matplotlib.pyplot as plt
import time

# 从自定义地图模块引入 25x25 矩阵及起终点
from map import MAP_GRID_25X25, START_POS, END_POS

class BiBFSGridEnv:
    def __init__(self, grid_matrix, start, end):
        self.grid = np.array(grid_matrix)
        self.rows, self.cols = self.grid.shape
        self.start = start
        self.end = end
        
    def get_neighbors(self, pos):
        r, c = pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr, nc] == 0:
                neighbors.append((nr, nc))
        return neighbors

    def run_bibfs(self):
        """双向广度优先搜索 (Bi-directional BFS)"""
        # 正向搜索初始化 (从 Start 开始)
        queue_f = [self.start]
        visited_f = {self.start: None}
        
        # 逆向搜索初始化 (从 End 开始)
        queue_b = [self.end]
        visited_b = {self.end: None}
        
        expansion_order = {}
        order_counter = 1
        intersect_node = None
        
        while queue_f and queue_b:
            # 1. 扩展正向搜索树的一个节点
            curr_f = queue_f.pop(0)
            if curr_f not in expansion_order:
                expansion_order[curr_f] = order_counter
                order_counter += 1
                
            # 检测是否与反向搜索相遇
            if curr_f in visited_b:
                intersect_node = curr_f
                break
                
            for neighbor in self.get_neighbors(curr_f):
                if neighbor not in visited_f:
                    visited_f[neighbor] = curr_f
                    queue_f.append(neighbor)
                    
            # 2. 扩展逆向搜索树的一个节点
            curr_b = queue_b.pop(0)
            if curr_b not in expansion_order:
                expansion_order[curr_b] = order_counter
                order_counter += 1
                
            # 检测是否与正向搜索相遇
            if curr_b in visited_f:
                intersect_node = curr_b
                break
                
            for neighbor in self.get_neighbors(curr_b):
                if neighbor not in visited_b:
                    visited_b[neighbor] = curr_b
                    queue_b.append(neighbor)
                    
        # 路径重建与拼接
        path = []
        if intersect_node is not None:
            # 从相遇点向正向起点(Start)回溯
            curr = intersect_node
            path_f = []
            while curr is not None:
                path_f.append(curr)
                curr = visited_f[curr]
            path_f.reverse()  # 翻转使其从 Start 到相遇点
            
            # 从相遇点向逆向终点(End)回溯
            curr = visited_b[intersect_node]
            path_b = []
            while curr is not None:
                path_b.append(curr)
                curr = visited_b[curr]
                
            # 拼接正反两条轨迹
            path = path_f + path_b
            
        return expansion_order, path

    def draw_heatmap(self, expansion_order, path, save_name):
        """绘制符合 25x25 地图学术规范的双向 BFS 收敛热力图"""
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
            ax.plot(path_x, path_y, color='lime', linewidth=4, label='Bi-BFS Shortest Path', zorder=5)
            
        ax.text(self.start[1]+0.5, self.start[0]+0.5, 'S', va='center', ha='center', color='white', weight='bold', fontsize=14, bbox=dict(facecolor='red', edgecolor='none'))
        ax.text(self.end[1]+0.5, self.end[0]+0.5, 'E', va='center', ha='center', color='black', weight='bold', fontsize=14, bbox=dict(facecolor='lightgray', edgecolor='none'))
        
        ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xticks(range(self.cols+1))
        ax.set_yticks(range(self.rows+1))
        ax.set_title("Bi-directional BFS Node Expansion Heatmap (25x25 Grid)", fontsize=14, pad=15)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Expansion Sequence (Steps)', rotation=270, labelpad=15, fontsize=11)
        
        plt.legend(loc='upper right', fontsize=11)
        plt.tight_layout()
        plt.savefig(save_name, dpi=300)
        print(f"📷 Bi-BFS 结果已成功渲染并保存至: {save_name}")
        plt.show()

if __name__ == "__main__":
    env = BiBFSGridEnv(MAP_GRID_25X25, START_POS, END_POS)
    
    print("🚀 开始执行双向广度优先搜索 (Bi-BFS)...")
    t_start = time.perf_counter()
    bibfs_order, bibfs_path = env.run_bibfs()
    t_end = time.perf_counter()
    
    execution_time_ms = (t_end - t_start) * 1000
    path_len = len(bibfs_path) - 1 if bibfs_path else 0
    visited_nodes = len(bibfs_order)
    
    print("\n" + "="*50)
    print(f"{'Bi-BFS 算法 25x25 地图量化报告':^40}")
    print("="*50)
    print(f"最终是否成功抵达终点: {len(bibfs_path) > 0}")
    print(f"规划最短路径步数:     {path_len} 步")
    print(f"算法总扩展节点数:     {visited_nodes} 个")
    print(f"算法纯求解耗时:       {execution_time_ms:.4f} ms")
    print("="*50 + "\n")
    
    env.draw_heatmap(bibfs_order, bibfs_path, "bibfs_25x25_heatmap.png")