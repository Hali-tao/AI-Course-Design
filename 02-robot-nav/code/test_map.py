import numpy as np
import matplotlib.pyplot as plt
import random

class AdvancedMapGenerator:
    def __init__(self, size=25):
        self.size = size
        
    def check_connectivity(self, grid, start, end):
        """使用 BFS 检查生成的随机地图是否至少有一条活路"""
        queue = [start]
        visited = {start}
        rows, cols = grid.shape
        
        while queue:
            curr = queue.pop(0)
            if curr == end:
                return True
            r, c = curr
            # 上下左右 4 方向探路
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return False

    def generate_random_hard_map(self, obstacle_density=0.20):
        """
        生成带有固定高难度 U 型陷阱 + 外围随机障碍物的地图
        :param obstacle_density: 外围开阔区域随机撒障碍物的密度 (20%)
        """
        while True:
            # 初始化全 0 地图
            grid = np.zeros((self.size, self.size))
            
            # --- 1. 固定核心骨架：大型 U 型口袋陷阱（开口朝左，底朝右） ---
            grid[7:18, 16] = 1  # 口袋底部
            grid[7, 8:17] = 1   # 口袋上壁
            grid[17, 8:17] = 1  # 口袋下壁
            
            # --- 2. 在外围区域随机生成离散障碍物 ---
            for r in range(self.size):
                for c in range(self.size):
                    # 口袋内部区域 (Row 8~16, Col 8~15) 和边界保持干净，确保机器人能从口袋走出来
                    if 7 <= r <= 17 and 7 <= c <= 17:
                        continue
                    # 预留起点和终点周围的一圈为空白，防止直接被出生点杀
                    if max(abs(r - 12), abs(c - 12)) <= 2: continue
                    if max(abs(r - 12), abs(c - 23)) <= 2: continue
                    
                    # 按概率随机撒下零散障碍物，模拟高随机碎石/未知掩体环境
                    if random.random() < obstacle_density:
                        grid[r, c] = 1
            
            # 设定固定的起点和终点
            start_pos = (12, 12)  # 口袋最深处
            end_pos = (12, 23)    # 口袋右侧外围
            
            # --- 3. 验证连通性 ---
            if self.check_connectivity(grid, start_pos, end_pos):
                return grid, start_pos, end_pos

    def draw_map(self, grid, start, end):
        rows, cols = grid.shape
        display_img = np.ones((rows, cols, 3))
        display_img[grid == 1] = [0.2, 0.4, 0.8]  # 蓝色障碍物
        
        plt.figure(figsize=(9, 9))
        plt.imshow(display_img, extent=[0, cols, rows, 0])
        
        # 标注起点 S 和 终点 E
        plt.text(start[1]+0.5, start[0]+0.5, 'S', va='center', ha='center', color='white', weight='bold', fontsize=14, bbox=dict(facecolor='red', edgecolor='none'))
        plt.text(end[1]+0.5, end[0]+0.5, 'E', va='center', ha='center', color='black', weight='bold', fontsize=14, bbox=dict(facecolor='lightgray', edgecolor='none'))
        
        plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
        plt.xticks(range(cols+1))
        plt.yticks(range(rows+1))
        plt.title("25 x 25 Randomized High-Difficulty Map (Verified)", fontsize=14, pad=15)
        
        plt.tight_layout()
        plt.savefig('random_advanced_map_25x25.png', dpi=300)
        print("🎉 随机高难度地图生成成功！已保存为 'random_advanced_map_25x25.png'。")
        print(f"起点 S: {start}, 终点 E: {end}")
        plt.show()

if __name__ == "__main__":
    # 实例化生成器
    generator = AdvancedMapGenerator(size=25)
    # 生成随机密度为 22% 的高难度地图
    grid, start, end = generator.generate_random_hard_map(obstacle_density=0.22)
    generator.draw_map(grid, start, end)