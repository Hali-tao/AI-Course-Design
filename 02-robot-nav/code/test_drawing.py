import numpy as np
import matplotlib.pyplot as plt
import heapq

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
                    
        # 重构路径
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
            
        # 优先队列元素格式: (f_score, count, current_node)
        # count 用于在 f_score 相同时打破僵局
        count = 0
        open_set = [(0, count, self.start)]
        came_from = {self.start: None}
        g_score = {self.start: 0}
        
        expansion_order = {}
        order_counter = 1
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            
            # 如果节点已经被处理过，跳过
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
                    
        # 重构路径
        path = []
        curr = self.end
        if curr in came_from:
            while curr is not None:
                path.append(curr)
                curr = came_from[curr]
            path.reverse()
        return expansion_order, path

    def draw_academic_heatmap(self, expansion_order, path, title):
        """绘制学术报告专用的扩展顺序热力图"""
        # 初始化一个用于热力图背景的矩阵，默认填满 NaN (在 plt 中显示为背景色)
        heatmap_matrix = np.full((self.rows, self.cols), np.nan)
        
        # 填充扩展顺序
        for (r, c), order in expansion_order.items():
            heatmap_matrix[r, c] = order
            
        # 创建显示图像 (RGB)，处理障碍物
        display_img = np.ones((self.rows, self.cols, 3))
        display_img[self.grid == 1] = [0.2, 0.4, 0.8]  # 蓝色障碍物
        
        plt.imshow(display_img, extent=[0, self.cols, self.rows, 0])
        
        # 绘制热力图层：YlOrRd (黄-橙-红) 
        # 颜色越深（红）表示越早被扩展，颜色越浅（黄）表示越晚被扩展
        # 刚好呼应了从起点（深红）向外扩散的过程
        vmax = max(expansion_order.values()) if expansion_order else 1
        masked_heatmap = np.ma.masked_where(np.isnan(heatmap_matrix), heatmap_matrix)
        im = plt.imshow(masked_heatmap, cmap='YlOrRd_r', extent=[0, self.cols, self.rows, 0], vmin=1, vmax=vmax, alpha=0.8)
        
        # 在格子里写上具体的数字序号，让报告更严谨
        for (r, c), order in expansion_order.items():
            if (r, c) != self.start and (r, c) != self.end:
                plt.text(c + 0.5, r + 0.5, str(order), va='center', ha='center', color='black', fontsize=9, weight='bold')

        # 绘制最终的最短路径
        if path:
            path_x = [c + 0.5 for r, c in path]
            path_y = [r + 0.5 for r, c in path]
            plt.plot(path_x, path_y, color='lime', linewidth=3.5, label='Shortest Path', linestyle='-', zorder=5) # 荧光绿路径更显眼
            
        # 标出起点 S 和 终点 E
        plt.text(self.start[1]+0.5, self.start[0]+0.5, 'S', va='center', ha='center', color='white', weight='bold', fontsize=12, bbox=dict(facecolor='red', edgecolor='none'))
        plt.text(self.end[1]+0.5, self.end[0]+0.5, 'E', va='center', ha='center', color='black', weight='bold', fontsize=12, bbox=dict(facecolor='lightgray', edgecolor='none'))
        
        plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
        plt.xticks(range(self.cols+1))
        plt.yticks(range(self.rows+1))
        plt.title(title, fontsize=14, pad=10)
        
        # 添加颜色条，用以解释颜色深浅的含义
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('Expansion Order (Step)', rotation=270, labelpad=15)


# --- 测试、数据统计与生成图表 ---
if __name__ == "__main__":
    import time

    # 复刻 PDF 场景图 (0通路，1障碍)
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
    start_pos = (2, 3)  # 对应红色 S 点
    end_pos = (4, 7)    # 对应灰色 E 点

    env = AcademicGridEnv(homework_matrix, start_pos, end_pos)
    
    # ------------------ 1. 运行并统计 BFS ------------------
    start_time = time.perf_counter()
    bfs_order, bfs_path = env.run_bfs()
    end_time = time.perf_counter()
    
    bfs_time_ms = (end_time - start_time) * 1000  # 转换为毫秒
    bfs_path_len = len(bfs_path) - 1 if bfs_path else 0 # 路径长度通常算移动步数（节点数-1）
    bfs_visited_count = len(bfs_order)

    # 绘制并保存 BFS
    plt.figure(figsize=(8, 5))
    env.draw_academic_heatmap(bfs_order, bfs_path, "BFS Algorithm Node Expansion Heatmap")
    plt.tight_layout()
    plt.savefig('bfs_heatmap.png', dpi=300)
    plt.close() # 关掉画布，防止内存占用
    
    # ------------------ 2. 运行并统计 A* ------------------
    start_time = time.perf_counter()
    astar_order, astar_path = env.run_astar()
    end_time = time.perf_counter()
    
    astar_time_ms = (end_time - start_time) * 1000
    astar_path_len = len(astar_path) - 1 if astar_path else 0
    astar_visited_count = len(astar_order)

    # 绘制并保存 A*
    plt.figure(figsize=(8, 5))
    env.draw_academic_heatmap(astar_order, astar_path, "A* Algorithm Node Expansion Heatmap")
    plt.tight_layout()
    plt.savefig('astar_heatmap.png', dpi=300)
    plt.close()

    # ------------------ 3. 打印学术对比表格 ------------------
    print("\n" + "="*50)
    print(f"{'算法性能量化对比数据 (Quantitative Results)':^50}")
    print("="*50)
    print(f"{'评估指标 (Metrics)':<25} | {'BFS 算法':<10} | {'A* 算法':<10}")
    print("-"*50)
    print(f"{'最终路径长度 (Path Length)':<25} | {bfs_path_len:<10} | {astar_path_len:<10}")
    print(f"{'总扩展节点数 (Visited Nodes)':<25} | {bfs_visited_count:<10} | {astar_visited_count:<10}")
    print(f"{'求解运行时间 (Time Cost)':<25} | {bfs_time_ms:.4f} ms  | {astar_time_ms:.4f} ms")
    print("="*50)
    print("🎉 统计完成！热力图图片已成功保存在当前代码目录下。\n")
