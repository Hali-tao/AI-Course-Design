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
    def __init__(self, env, pop_size=150, max_gen=1000, chromosome_len=12, mutate_rate=0.2):
        self.env = env
        self.pop_size = pop_size          # 种群大小
        self.max_gen = max_gen            # 最大进化代数
        self.gene_len = chromosome_len    # 染色体长度（中间关键拐点的数量）
        self.mutate_rate = mutate_rate    # 变异概率
        
    def create_random_chromosome(self):
        """随机生成一条染色体：包含一系列在地图内的随机关键拐点坐标"""
        chromosome = []
        for _ in range(self.gene_len):
            r = random.randint(0, self.env.rows - 1)
            c = random.randint(0, self.env.cols - 1)
            chromosome.append((r, c))
        return chromosome

    def initialize_population(self):
        return [self.create_random_chromosome() for _ in range(self.pop_size)]

    def decode_to_full_path(self, chromosome):
        """将染色体的关键拐点用‘连线’的方式解码成完整的网格移动路径（严格4方向限制）"""
        full_path = [self.env.start]
        curr = self.env.start
        
        waypoints = list(chromosome) + [self.env.end]
        
        for target in waypoints:
            tr, tc = target
            
            while curr != target:
                r, c = curr
                
                if r != tr:
                    dr = 1 if tr > r else -1
                    curr = (r + dr, c)
                elif c != tc:
                    dc = 1 if tc > c else -1
                    curr = (r, c + dc)
                
                full_path.append(curr)
                
                if len(full_path) > 400:  
                    return full_path
        return full_path

    def calculate_fitness(self, chromosome):
        """适应度函数：核心优胜劣汰依据"""
        full_path = self.decode_to_full_path(chromosome)
        
        collision_count = 0
        length_penalty = len(full_path)
        
        for node in full_path:
            if not self.env.is_valid(node):
                collision_count += 1
                
        final_node = full_path[-1]
        dist_to_end = abs(final_node[0] - self.env.end[0]) + abs(final_node[1] - self.env.end[1])
        
        # 适应度公式
        fitness = 10000.0 - (collision_count * 600.0) - (length_penalty * 15.0) - (dist_to_end * 400.0)
        return max(0.1, fitness)

    def selection(self, pop, fitnesses):
        total_fit = sum(fitnesses)
        probs = [f / total_fit for f in fitnesses]
        return random.choices(pop, weights=probs, k=2)

    def crossover(self, parent1, parent2):
        if random.random() < 0.8:
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

    def evolve(self):
        population = self.initialize_population()
        best_fitness_history = []
        
        best_chromo = None
        best_fit = -float('inf')
        
        print("🧬 遗传算法(GA)开始种群迭代进化(共1000代)...")
        
        for gen in range(self.max_gen):
            fitnesses = [self.calculate_fitness(ch) for ch in population]
            
            max_idx = np.argmax(fitnesses)
            current_best_fit = fitnesses[max_idx]
            best_fitness_history.append(current_best_fit)
            
            if current_best_fit > best_fit:
                best_fit = current_best_fit
                best_chromo = population[max_idx].copy()
                
            next_pop = []
            next_pop.append(best_chromo.copy())
            next_pop.append(population[max_idx].copy())
            
            while len(next_pop) < self.pop_size:
                p1, p2 = self.selection(population, fitnesses)
                c1, c2 = self.crossover(p1, p2)
                self.mutate(c1)
                self.mutate(c2)
                next_pop.extend([c1, c2])
                
            population = next_pop[:self.pop_size]
            
            if (gen + 1) % 50 == 0:
                print(f"世代: {gen+1:3d} / {self.max_gen} | 当前代最高适应度: {current_best_fit:.2f}")
                
        print("🎉 GA进化结束！已提取全局最优染色体。")
        return best_chromo, best_fitness_history

def test_and_draw_ga(env, planner, best_chromo):
    t_start = time.perf_counter()
    full_path = planner.decode_to_full_path(best_chromo)
    t_end = time.perf_counter()
    
    # 提取撞墙前的合法路径段
    valid_path = []
    for node in full_path:
        if env.is_valid(node):
            if not valid_path or node != valid_path[-1]:
                valid_path.append(node)
        else:
            # 撞墙时，保留撞墙前的最后一步，并终止，以便观察卡在哪里
            if not valid_path or node != valid_path[-1]:
                valid_path.append(node)
            break
            
    execution_time_ms = (t_end - t_start) * 1000
    is_success = (len(valid_path) > 0 and valid_path[-1] == env.end)
    
    print("\n" + "="*50)
    print(f"{'遗传算法 (GA) 最终量化指标':^30}")
    print("="*50)
    print(f"最终是否成功抵达终点: {is_success}")
    if is_success:
        print(f"算法生成的最优路径步数: {len(valid_path) - 1} 步")
    else:
        print(f"⚠️ 算法未收敛到终点！当前尝试路径长度: {len(valid_path)} 步 (最终以撞墙/无法前进告终)")
    print(f"单次路径生成测试耗时:    {execution_time_ms:.4f} ms")
    print("="*50 + "\n")
    
    # 绘图
    display_img = np.ones((env.rows, env.cols, 3))
    display_img[env.grid == 1] = [0.2, 0.4, 0.8] # 蓝色障碍物
    
    plt.figure(figsize=(10, 10))
    plt.imshow(display_img, extent=[0, env.cols, env.rows, 0])
    
    # 🎨 新增功能：不管收敛与否，只要有路径数据就绘制
    if len(valid_path) > 0:
        path_x = [c + 0.5 for r, c in valid_path]
        path_y = [r + 0.5 for r, c in valid_path]
        
        # 成功用洋红色（magenta），失败用橘红色（orangered）以示区分
        path_color = 'magenta' if is_success else 'orangered'
        path_label = 'GA Optimal Path' if is_success else 'GA Unconverged Current Best Path'
        
        plt.plot(path_x, path_y, color=path_color, linewidth=4, label=path_label, zorder=5)
        
    plt.text(env.start[1]+0.5, env.start[0]+0.5, 'S', va='center', ha='center', color='white', weight='bold', fontsize=14, bbox=dict(facecolor='red', edgecolor='none'))
    plt.text(env.end[1]+0.5, env.end[0]+0.5, 'E', va='center', ha='center', color='black', weight='bold', fontsize=14, bbox=dict(facecolor='lightgray', edgecolor='none'))
    
    plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
    plt.xticks(range(env.cols+1))
    plt.yticks(range(env.rows+1))
    
    # 动态调整标题
    title_str = "GA Pathfinding Strategy (SUCCESS)" if is_success else "GA Pathfinding Strategy (FAILED - Current Attempt)"
    plt.title(title_str, fontsize=12, pad=15, color='black' if is_success else 'red')
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("ga_path_result.png", dpi=300)
    plt.show()
    return is_success, valid_path

if __name__ == "__main__":
    ga_env = GAGridEnv(MAP_GRID_25X25, START_POS, END_POS)
    planner = GeneticAlgorithmPathplanner(ga_env)
    
    best_chromo, fit_history = planner.evolve()
    
    # 绘制GA收敛曲线
    plt.figure(figsize=(8, 4))
    plt.plot(fit_history, color='magenta', linewidth=2)
    plt.title("Genetic Algorithm Fitness Convergence Curve")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig("ga_convergence_curve.png", dpi=300)
    plt.show()
    
    test_and_draw_ga(ga_env, planner, best_chromo)