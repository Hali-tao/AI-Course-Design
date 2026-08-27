import numpy as np
import matplotlib.pyplot as plt
import heapq
import time

# 引入精准还原的 25x25 地图数据与起终点
from map import MAP_GRID_25X25, START_POS, END_POS

class ComprehensiveGridEnv:
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

    def run_bfs(self):
        """1. 广度优先搜索 (BFS)"""
        queue = [self.start]
        visited = {self.start: None}
        expansion_order = {}  
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

    def run_bibfs(self):
        """2. 双向广度优先搜索 (Bi-directional BFS)"""
        queue_f = [self.start]
        visited_f = {self.start: None}
        queue_b = [self.end]
        visited_b = {self.end: None}
        
        expansion_order = {}
        order_counter = 1
        intersect_node = None
        
        while queue_f and queue_b:
            # 扩展正向树
            curr_f = queue_f.pop(0)
            if curr_f not in expansion_order:
                expansion_order[curr_f] = order_counter
                order_counter += 1
            if curr_f in visited_b:
                intersect_node = curr_f
                break
            for neighbor in self.get_neighbors(curr_f):
                if neighbor not in visited_f:
                    visited_f[neighbor] = curr_f
                    queue_f.append(neighbor)
            # 扩展逆向树
            curr_b = queue_b.pop(0)
            if curr_b not in expansion_order:
                expansion_order[curr_b] = order_counter
                order_counter += 1
            if curr_b in visited_f:
                intersect_node = curr_b
                break
            for neighbor in self.get_neighbors(curr_b):
                if neighbor not in visited_b:
                    visited_b[neighbor] = curr_b
                    queue_b.append(neighbor)
        path = []
        if intersect_node is not None:
            curr = intersect_node
            path_f = []
            while curr is not None:
                path_f.append(curr)
                curr = visited_f[curr]
            path_f.reverse()
            curr = visited_b[intersect_node]
            path_b = []
            while curr is not None:
                path_b.append(curr)
                curr = visited_b[curr]
            path = path_f + path_b
        return expansion_order, path

    def run_dijkstra(self):
        """3. Dijkstra 算法"""
        count = 0
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

    def run_astar(self):
        """4. A* 算法 (使用曼哈顿距离)"""
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

    def draw_subplot_heatmap(self, ax, expansion_order, path, title):
        """在指定的子图(ax)上绘制25x25大地图学术热力图"""
        heatmap_matrix = np.full((self.rows, self.cols), np.nan)
        for (r, c), order in expansion_order.items():
            heatmap_matrix[r, c] = order
            
        display_img = np.ones((self.rows, self.cols, 3))
        display_img[self.grid == 1] = [0.2, 0.4, 0.8]  # 蓝色障碍物
        
        ax.imshow(display_img, extent=[0, self.cols, self.rows, 0])
        
        vmax = max(expansion_order.values()) if expansion_order else 1
        masked_heatmap = np.ma.masked_where(np.isnan(heatmap_matrix), heatmap_matrix)
        
        # 采用 YlOrRd_r 颜色条渲染
        im = ax.imshow(masked_heatmap, cmap='YlOrRd_r', extent=[0, self.cols, self.rows, 0], vmin=1, vmax=vmax, alpha=0.8)
        
        if path:
            path_x = [c + 0.5 for r, c in path]
            path_y = [r + 0.5 for r, c in path]
            ax.plot(path_x, path_y, color='lime', linewidth=3, label='Shortest Path', zorder=5)
            
        ax.text(self.start[1]+0.5, self.start[0]+0.5, 'S', va='center', ha='center', color='white', weight='bold', fontsize=11, bbox=dict(facecolor='red', edgecolor='none'))
        ax.text(self.end[1]+0.5, self.end[0]+0.5, 'E', va='center', ha='center', color='black', weight='bold', fontsize=11, bbox=dict(facecolor='lightgray', edgecolor='none'))
        
        ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xticks(range(self.cols+1))
        ax.set_yticks(range(self.rows+1))
        # 隐藏密集网格坐标标签以保持学术大图美观
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(title, fontsize=12, pad=10)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)


# --- 自动化一键流水线测试 ---
if __name__ == "__main__":
    env = ComprehensiveGridEnv(MAP_GRID_25X25, START_POS, END_POS)
    
    print("⏳ 四大路径规划算法联调流水线启动...")
    
    # 1. 运行并统计 BFS
    t0 = time.perf_counter()
    bfs_order, bfs_path = env.run_bfs()
    t1 = time.perf_counter()
    bfs_time = (t1 - t0) * 1000
    
    # 2. 运行并统计 双向 BFS
    t0 = time.perf_counter()
    bibfs_order, bibfs_path = env.run_bibfs()
    t1 = time.perf_counter()
    bibfs_time = (t1 - t0) * 1000
    
    # 3. 运行并统计 Dijkstra
    t0 = time.perf_counter()
    dijk_order, dijk_path = env.run_dijkstra()
    t1 = time.perf_counter()
    dijk_time = (t1 - t0) * 1000
    
    # 4. 运行并统计 A*
    t0 = time.perf_counter()
    astar_order, astar_path = env.run_astar()
    t1 = time.perf_counter()
    astar_time = (t1 - t0) * 1000
    
    print("🎉 算法计算完毕，正在渲染 2x2 学术矩阵组合图...")

    # 创建 2x2 的画幅拼接
    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    
    env.draw_subplot_heatmap(axes[0, 0], bfs_order, bfs_path, "A) Breadth-First Search (BFS) Heatmap")
    env.draw_subplot_heatmap(axes[0, 1], bibfs_order, bibfs_path, "B) Bi-directional BFS Heatmap")
    env.draw_subplot_heatmap(axes[1, 0], dijk_order, dijk_path, "C) Dijkstra Algorithm Heatmap")
    env.draw_subplot_heatmap(axes[1, 1], astar_order, astar_path, "D) A* Algorithm (Manhattan) Heatmap")
    
    plt.suptitle("Comparative Analysis of Node Expansion Space Patterns", fontsize=15, weight='bold', y=0.98)
    plt.tight_layout()
    
    # 保存高分辨率组合图
    save_fig_name = "traditional_algorithms_4in1_heatmap.png"
    plt.savefig(save_fig_name, dpi=300, bbox_inches='tight')
    print(f"📷 四合一矩阵对比图已保存至: {save_fig_name}")
    plt.show()

    # 打印学术量化指标表格数据
    def get_len(p): return len(p) - 1 if p else 0
    print("\n" + "="*70)
    print(f"{'学术报告专用：四种传统与改良算法量化性能比对表':^60}")
    print("="*70)
    print(f"{'评估指标 (Metrics)':<25} | {'BFS':<8} | {'双向 BFS':<8} | {'Dijkstra':<8} | {'A* 算法':<8}")
    print("-"*70)
    print(f"{'路径总步数 (Path Steps)':<25} | {get_len(bfs_path):<8} | {get_len(bibfs_path):<8} | {get_len(dijk_path):<8} | {get_len(astar_path):<8}")
    print(f"{'总扩展节点数 (Visited Nodes)':<25} | {len(bfs_order):<8} | {len(bibfs_order):<8} | {len(dijk_order):<8} | {len(astar_order):<8}")
    print(f"{'纯求解时间 (Time Cost)':<25} | {bfs_time:.3f}ms | {bibfs_time:.3f}ms   | {dijk_time:.3f}ms  | {astar_time:.3f}ms")
    print("="*70 + "\n")