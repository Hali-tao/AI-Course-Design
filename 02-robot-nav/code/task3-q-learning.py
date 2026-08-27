import numpy as np
import matplotlib.pyplot as plt
import random
import time

# 引入之前精准还原的地图数据与起终点
from map import MAP_GRID_25X25, START_POS, END_POS

class QLearningGridEnv:
    def __init__(self, grid_matrix, start, end):
        self.grid = np.array(grid_matrix)
        self.rows, self.cols = self.grid.shape
        self.start = start
        self.end = end
        
        # 动作空间：0=上, 1=下, 2=左, 3=右 (使用4方向使离散动作状态更稳定)
        self.action_space = [0, 1, 2, 3]
        self.action_effects = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1)    # 右
        }
        
    def reset(self):
        self.current_state = self.start
        return self.current_state

    def step(self, action):
        """执行动作，返回：新状态, 奖励值, 是否结束"""
        dr, dc = self.action_effects[action]
        next_r = self.current_state[0] + dr
        next_c = self.current_state[1] + dc
        next_state = (next_r, next_c)
        
        # 1. 边界与撞墙检测
        if not (0 <= next_r < self.rows and 0 <= next_c < self.cols) or self.grid[next_r, next_c] == 1:
            # 撞墙或越界：状态保持不变，给予巨大的负奖励
            reward = -15.0
            done = False
            return self.current_state, reward, done
            
        # 2. 正常移动
        self.current_state = next_state
        
        # 3. 抵达终点检测
        if self.current_state == self.end:
            reward = 200.0  # 极大的正奖励
            done = True
            return self.current_state, reward, done
            
        # 4. 普通移动的引导奖励（时间惩罚 + 距离启发项）
        # 计算新位置到终点的曼哈顿距离
        dist = abs(self.current_state[0] - self.end[0]) + abs(self.current_state[1] - self.end[1])
        # 距离越近，扣分越少，引导其向右走；每走一步扣1分，逼迫其找最短路径
        reward = -1.0 - 0.1 * dist
        done = False
        
        return self.current_state, reward, done


class QLearningAgent:
    def __init__(self, rows, cols, action_size=4, alpha=0.1, gamma=0.95, epsilon=0.3):
        self.rows = rows
        self.cols = cols
        self.action_size = action_size
        self.alpha = alpha      # 学习率
        self.gamma = gamma      # 折扣因子（重视长远奖励）
        self.epsilon = epsilon  # 探索率
        
        # 初始化 Q 表：维度为 (行, 列, 动作数)
        self.q_table = np.zeros((rows, cols, action_size))
        
    def choose_action(self, state):
        """ε-greedy 策略：平衡探索与利用"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)  # 随机探索未知世界
        else:
            return np.argmax(self.q_table[state[0], state[1]])  # 选择当前最优动作

    def learn(self, state, action, reward, next_state):
        """贝尔曼方程更新 Q 值"""
        q_predict = self.q_table[state[0], state[1], action]
        # 目标值：当前奖励 + 未来最大期望收益
        q_target = reward + self.gamma * np.max(self.q_table[next_state[0], next_state[1]])
        # 更新 Q 表
        self.q_table[state[0], state[1], action] += self.alpha * (q_target - q_predict)


def train_agent(env, agent, episodes=1500, max_steps_per_episode=300):
    """训练智能体"""
    print("🚀 强化学习模型开始训练(共1500轮自主试错进化)...")
    reward_history = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        # 随着训练进行，逐渐减小探索率，让智能体变得越来越笃定和聪明
        if episode > 0 and episode % 100 == 0:
            agent.epsilon = max(0.01, agent.epsilon * 0.85)
            
        for step in range(max_steps_per_episode):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            # 更新 Q 表
            agent.learn(state, action, reward, next_state)
            
            state = next_state
            episode_reward += reward
            if done:
                break
                
        reward_history.append(episode_reward)
        if (episode + 1) % 150 == 0:
            print(f"轮次: {episode+1:4d} / {episodes} | 当前探索率 ε: {agent.epsilon:.4f} | 本轮总收益: {episode_reward:.2f}")
            
    print("🎉 训练完成! Q 表已经完全收敛。")
    return reward_history


def test_and_draw_rl(env, agent, train_time_s):
    """测试训练好的 Q 表，提取最优寻径轨迹并绘制可视化结果"""
    state = env.reset()
    path = [state]
    visited = [state]
    
    max_steps = 200
    step = 0
    done = False
    
    # 路径规划（在线推理）阶段精密计时
    t_start = time.perf_counter()
    while not done and step < max_steps:
        # 测试时不进行随机探索，百分之百贪婪选择 Q 值最大的动作
        action = np.argmax(agent.q_table[state[0], state[1]])
        next_state, _, done = env.step(action)
        
        if next_state == state: # 如果陷入死循环说明没学好
            break
            
        state = next_state
        path.append(state)
        visited.append(state)
        step += 1
    t_end = time.perf_counter()
    
    execution_time_ms = (t_end - t_start) * 1000
    is_success = (path[-1] == env.end)
    
    print("\n" + "="*60)
    print(f"{'Q-Learning 强化学习在 25x25 地图下的最终量化指标':^40}")
    print("="*60)
    print(f"最终是否成功抵达终点: {is_success}")
    print(f"算法生成的最优路径步数: {len(path) - 1} 步")
    print(f"📊 模型离线训练耗时:   {train_time_s:.4f} s (约 {train_time_s/60:.2f} 分钟)")
    print(f"⚡ 单次路径推理测试耗时: {execution_time_ms:.4f} ms")
    print(f"最终到达终点坐标:       {path[-1]}")
    print("="*60 + "\n")
    
    # 绘制寻径大图
    display_img = np.ones((env.rows, env.cols, 3))
    display_img[env.grid == 1] = [0.2, 0.4, 0.8]  # 蓝色障碍物
    
    plt.figure(figsize=(10, 10))
    plt.imshow(display_img, extent=[0, env.cols, env.rows, 0])
    
    # 绘制路径（荧光绿）
    if path:
        path_x = [c + 0.5 for r, c in path]
        path_y = [r + 0.5 for r, c in path]
        plt.plot(path_x, path_y, color='lime', linewidth=4, label='RL Optimal Path', zorder=5)
        
    plt.text(env.start[1]+0.5, env.start[0]+0.5, 'S', va='center', ha='center', color='white', weight='bold', fontsize=14, bbox=dict(facecolor='red', edgecolor='none'))
    plt.text(env.end[1]+0.5, env.end[0]+0.5, 'E', va='center', ha='center', color='black', weight='bold', fontsize=14, bbox=dict(facecolor='lightgray', edgecolor='none'))
    
    plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
    plt.xticks(range(env.cols+1))
    plt.yticks(range(env.rows+1))
    plt.title("Q-Learning Reinforcement Learning Pathfinding Success (No Potential Fields)", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig("q_learning_success.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # 实例化环境与智能体
    rl_env = QLearningGridEnv(MAP_GRID_25X25, START_POS, END_POS)
    rl_agent = QLearningAgent(rows=rl_env.rows, cols=rl_env.cols)
    
    # ⏱️ 核心改动：针对强化学习 1500 轮训练进行高精度计时
    train_start_time = time.perf_counter()
    rewards = train_agent(rl_env, rl_agent, episodes=1500)
    train_end_time = time.perf_counter()
    
    total_train_time_s = train_end_time - train_start_time
    
    # 绘制训练收敛曲线（用于实验报告的绝佳学术图表）
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, color='darkorange', alpha=0.6)
    plt.title("Q-Learning Training Reward Convergence Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig("rl_convergence_curve.png", dpi=300)
    plt.show()
    
    # 测试最终的最优路径，并将训练耗时传入报告展示
    test_and_draw_rl(rl_env, rl_agent, total_train_time_s)