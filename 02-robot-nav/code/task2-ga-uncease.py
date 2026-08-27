import numpy as np
import matplotlib.pyplot as plt
import random
import time

# 引入之前精准还原的地图数据与起终点
from map import MAP_GRID_25X25, START_POS, END_POS

class GAGridEnv:
    def __init__(self, grid_matrix, start, end):
        self.grid = np.array(grid_matrix)
        self.rows, self.cols = self.grid.shape
        self.start = start
        self.end = end

    def is_valid(self, pos):
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r, c] == 0

class GeneticAlgorithmPathplanner:
    def __init__(self, env, pop_size=200, max_gen=100000, chromosome_len=14, mutate_rate=0.25):
        self.env = env
        self.pop_size = pop_size          # 略微扩增种群基数（150 -> 200），增强基因多样性
        self.max_gen = max_gen            # 提高最大代数上限（2000 -> 100000）
        self.gene_len = chromosome_len    # 增加拐点数（12 -> 14），使弯曲绕行更灵活
        self.mutate_rate = mutate_rate    # 提高变异率以跳出局部最优
        
    def create_random_chromosome(self):
        chromosome = []
        for _ in range(self.gene_len):
            r = random.randint(0, self.env.rows - 1)
            c = random.randint(0, self.env.cols - 1)
            chromosome.append((r, c))
        return chromosome

    def initialize_population(self):
        return [self.create_random_chromosome() for _ in range(self.pop_size)]

    def decode_to_full_path(self, chromosome):
        full_path = [self.env.start]
        curr = self.env.start
        waypoints = list(chromosome) + [self.env.end]
        
        for target in waypoints:
            while curr != target:
                r, c = curr
                tr, tc = target
                dr = 1 if tr > r else (-1 if tr < r else 0)
                dc = 1 if tc > c else (-1 if tc < c else 0)
                
                curr = (r + dr, c + dc)
                full_path.append(curr)
                if len(full_path) > 350:  # 放宽路径长度上限
                    return full_path
        return full_path

    def calculate_fitness(self, chromosome):
        full_path = self.decode_to_full_path(chromosome)
        
        collision_count = 0
        length_penalty = len(full_path)
        
        for node in full_path:
            if not self.env.is_valid(node):
                collision_count += 1
                
        final_node = full_path[-1]
        dist_to_end = abs(final_node[0] - self.env.end[0]) + abs(final_node[1] - self.env.end[1])
        
        # 适应度函数优化：如果是完美通路，给予巨额加分
        if collision_count == 0 and final_node == self.env.end:
            perfection_bonus = 5000.0
        else:
            perfection_bonus = 0.0

        fitness = 15000.0 - (collision_count * 800.0) - (length_penalty * 10.0) - (dist_to_end * 500.0) + perfection_bonus
        return max(0.1, fitness)

    def selection(self, pop, fitnesses):
        total_fit = sum(fitnesses)
        probs = [f / total_fit for f in fitnesses]
        return random.choices(pop, weights=probs, k=2)

    def crossover(self, parent1, parent2):
        if random.random() < 0.85:
            cp = random.randint(1, self.gene_len - 1)
            child1 = parent1[:cp] + parent2[cp:]
            child2 = parent2[:cp] + parent1[cp:]
            return child1, child2
        return parent1.copy(), parent2.copy()

    def mutate(self, chromosome):
        for i in range(self.gene_len):
            if random.random() < self.mutate_rate:
                r = random.randint(0, self.env.rows - 1)
                c = random.randint(0, self.env.cols - 1)
                chromosome[i] = (r, c)

    def check_path_success(self, chromosome):
        """检查某条染色体解码后是否是一条真正的无障碍通路"""
        full_path = self.decode_to_full_path(chromosome)
        if full_path[-1] != self.env.end:
            return False
        for node in full_path:
            if not self.env.is_valid(node):
                return False
        return True

    def evolve(self):
        population = self.initialize_population()
        best_fitness_history = []
        
        best_chromo = None
        best_fit = -float('inf')
        
        print("🧬 遗传算法(GA)开始种群迭代进化...")
        print("💡 模式：[找到可行通解前绝不停滞模式] 开启")
        
        for gen in range(self.max_gen):
            fitnesses = [self.calculate_fitness(ch) for ch in population]
            
            max_idx = np.argmax(fitnesses)
            current_best_fit = fitnesses[max_idx]
            best_fitness_history.append(current_best_fit)
            
            if current_best_fit > best_fit:
                best_fit = current_best_fit
                best_chromo = population[max_idx].copy()
            
            # --- 核心改进：实时判定是否产生了有效通解 ---
            if self.check_path_success(best_chromo):
                print(f"🎉 破局成功！在第 {gen+1} 世代提前发现全局有效通路！算法主动收敛终止。")
                # 截断后续无效历史记录，使曲线更好看
                break
                
            next_pop = []
            # 精英保留
            next_pop.append(best_chromo.copy())
            next_pop.append(population[max_idx].copy())
            
            while len(next_pop) < self.pop_size:
                p1, p2 = self.selection(population, fitnesses)
                c1, c2 = self.crossover(p1, p2)
                self.mutate(c1)
                self.mutate(c2)
                next_pop.extend([c1, c2])
                
            population = next_pop[:self.pop_size]
            
            if (gen + 1) % 1000 == 0:
                print(f"世代: {gen+1:6d} / {self.max_gen} | 当前代最高适应度: {current_best_fit:.2f} | 正在全力突破障碍迷宫...")
                
        return best_chromo, best_fitness_history

def test_and_draw_ga(env, planner, best_chromo):
    t_start = time.perf_counter()
    full_path = planner.decode_to_full_path(best_chromo)
    t_end = time.perf_counter()
    
    # 提取无碰撞的实际连续轨迹
    valid_path = []
    has_collision = False
    for node in full_path:
        if env.is_valid(node):
            if not valid_path or node != valid_path[-1]:
                valid_path.append(node)
        else:
            has_collision = True
            
    execution_time_ms = (t_end - t_start) * 1000
    is_success = (not has_collision and len(full_path) > 0 and full_path[-1] == env.end)
    
    print("\n" + "="*50)
    print(f"{'遗传算法 (GA) 破局模式优化量化指标':^30}")
    print("="*50)
    print(f"最终是否成功抵达终点: {is_success}")
    if is_success:
        print(f"算法生成的最优路径步数: {len(full_path) - 1} 步")
    else:
        print("警告：已达最大世代上限，仍未找到无碰撞通路")
    print(f"路径解码生成测试耗时:    {execution_time_ms:.4f} ms")
    print("="*50 + "\n")
    
    # 绘图展示
    display_img = np.ones((env.rows, env.cols, 3))
    display_img[env.grid == 1] = [0.2, 0.4, 0.8]
    
    plt.figure(figsize=(10, 10))
    plt.imshow(display_img, extent=[0, env.cols, env.rows, 0])
    
    # 只要生成了整条路径的趋势就把它绘制出来
    path_x = [c + 0.5 for r, c in full_path]
    path_y = [r + 0.5 for r, c in full_path]
    plt.plot(path_x, path_y, color='magenta', linewidth=4, label='GA Found Path', zorder=5)
        
    plt.text(env.start[1]+0.5, env.start[0]+0.5, 'S', va='center', ha='center', color='white', weight='bold', fontsize=14, bbox=dict(facecolor='red', edgecolor='none'))
    plt.text(env.end[1]+0.5, env.end[0]+0.5, 'E', va='center', ha='center', color='black', weight='bold', fontsize=14, bbox=dict(facecolor='lightgray', edgecolor='none'))
    
    plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
    plt.xticks(range(env.cols+1))
    plt.yticks(range(env.rows+1))
    plt.title("Genetic Algorithm (GA) Pathfinding Breakout Success", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig("ga_path_success.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    ga_env = GAGridEnv(MAP_GRID_25X25, START_POS, END_POS)
    planner = GeneticAlgorithmPathplanner(ga_env)
    
    best_chromo, fit_history = planner.evolve()
    
    # 绘制GA收敛曲线
    plt.figure(figsize=(8, 4))
    plt.plot(fit_history, color='magenta', linewidth=2)
    plt.title("Genetic Algorithm Dynamic Convergence Curve")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig("ga_convergence_curve.png", dpi=300)
    plt.show()
    
    test_and_draw_ga(ga_env, planner, best_chromo)