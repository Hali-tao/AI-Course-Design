import numpy as np
import matplotlib.pyplot as plt
import heapq
import time

class AcademicGridEnv:
    def __init__(self, grid_matrix, start, end):
        self.grid = np.array(grid_matrix)
        self.rows, self.cols = self.grid.shape
        self.start = start
        self.end = end
        
    def get_neighbors(self, pos):
        r, c = pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 上下左右
        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr, nc] == 0:
                neighbors.append((nr, nc))
        return neighbors

    def run_bfs(self):
        """广度优先搜索 (BFS)"""
        queue = [self.start]
        visited = {self.start: None}
        expansion_order = {}  # 记录扩展顺序
        order_counter = 1
        
        while queue:
            current = queue.pop(0)
            expansion_order[current] = order_counter
            order_counter += 1
            
            if current == self.end:
                break
                
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited[neighbor] = current
                    queue.append(neighbor)
                    
        path = []
        curr = self.end
        if curr in visited:
            while curr is not None:
                path.append(curr)
                curr = visited[curr]
            path.reverse()
        return expansion_order, path

    def run_astar(self):
        """A* 算法 (使用曼哈顿距离作为启发式函数)"""
        def heuristic(p1, p2):
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

    def draw_academic_heatmap(self, ax, expansion_order, path, title):
        """在指定的子图(ax)上绘制学术报告专用的扩展顺序热力图"""
        heatmap_matrix = np.full((self.rows, self.cols), np.nan)
        for (r, c), order in expansion_order.items():
            heatmap_matrix[r, c] = order
            
        display_img = np.ones((self.rows, self.cols, 3))
        display_img[self.grid == 1] = [0.2, 0.4, 0.8]  # 蓝色障碍物
        
        # 使用传入的 ax 对象进行绘制
        ax.imshow(display_img, extent=[0, self.cols, self.rows, 0])
        
        vmax = max(expansion_order.values()) if expansion_order else 1
        masked_heatmap = np.ma.masked_where(np.isnan(heatmap_matrix), heatmap_matrix)
        im = ax.imshow(masked_heatmap, cmap='YlOrRd_r', extent=[0, self.cols, self.rows, 0], vmin=1, vmax=vmax, alpha=0.8)
        
        for (r, c), order in expansion_order.items():
            if (r, c) != self.start and (r, c) != self.end:
                ax.text(c + 0.5, r + 0.5, str(order), va='center', ha='center', color='black', fontsize=9, weight='bold')

        if path:
            path_x = [c + 0.5 for r, c in path]
            path_y = [r + 0.5 for r, c in path]
            ax.plot(path_x, path_y, color='lime', linewidth=3.5, label='Shortest Path', zorder=5)
            
        ax.text(self.start[1]+0.5, self.start[0]+0.5, 'S', va='center', ha='center', color='white', weight='bold', fontsize=12, bbox=dict(facecolor='red', edgecolor='none'))
        ax.text(self.end[1]+0.5, self.end[0]+0.5, 'E', va='center', ha='center', color='black', weight='bold', fontsize=12, bbox=dict(facecolor='lightgray', edgecolor='none'))
        
        ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xticks(range(self.cols+1))
        ax.set_yticks(range(self.rows+1))
        ax.set_title(title, fontsize=12, pad=10)
        
        # 为当前子图单独添加颜色轴
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Expansion Order (Step)', rotation=270, labelpad=15, fontsize=9)


# --- 测试、数据统计与生成图表 ---
if __name__ == "__main__":
    homework_matrix = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]
    start_pos = (2, 3)
    end_pos = (4, 7)

    env = AcademicGridEnv(homework_matrix, start_pos, end_pos)
    
    # 1. 运行并统计 BFS
    start_time = time.perf_counter()
    bfs_order, bfs_path = env.run_bfs()
    end_time = time.perf_counter()
    bfs_time_ms = (end_time - start_time) * 1000
    bfs_path_len = len(bfs_path) - 1 if bfs_path else 0
    bfs_visited_count = len(bfs_order)
    
    # 2. 运行并统计 A*
    start_time = time.perf_counter()
    astar_order, astar_path = env.run_astar()
    end_time = time.perf_counter()
    astar_time_ms = (end_time - start_time) * 1000
    astar_path_len = len(astar_path) - 1 if astar_path else 0
    astar_visited_count = len(astar_order)

    # 3. 创建合并画布：1行2列，设置宽为15，高为5.5
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    
    # 左图画 BFS
    env.draw_academic_heatmap(axes[0], bfs_order, bfs_path, "BFS Algorithm Node Expansion Heatmap")
    
    # 右图画 A*
    env.draw_academic_heatmap(axes[1], astar_order, astar_path, "A* Algorithm Node Expansion Heatmap")
    
    # 保存组合图
    plt.tight_layout()
    plt.savefig('combined_heatmap.png', dpi=300) # 生成一张高分辨率的组合对比图
    print("🎉 左右对比图已成功保存为 'combined_heatmap.png'")
    plt.show()

    # 4. 打印学术对比表格
    print("\n" + "="*50)
    print(f"{'算法性能量化对比数据 (Quantitative Results)':^50}")
    print("="*50)
    print(f"{'评估指标 (Metrics)':<25} | {'BFS 算法':<10} | {'A* 算法':<10}")
    print("-"*50)
    print(f"{'最终路径长度 (Path Length)':<25} | {bfs_path_len:<10} | {astar_path_len:<10}")
    print(f"{'总扩展节点数 (Visited Nodes)':<25} | {bfs_visited_count:<10} | {astar_visited_count:<10}")
    print(f"{'求解运行时间 (Time Cost)':<25} | {bfs_time_ms:.4f} ms  | {astar_time_ms:.4f} ms")
    print("="*50 + "\n")