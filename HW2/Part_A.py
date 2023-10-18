import mlrose_hiive as mlrose
import numpy as np
import matplotlib.pyplot as plt
import six
weights = np.random.randint(1, 50, 50)  # 100个物品，重量随机在1到50之间
values = np.random.randint(1, 50, 50)  # 100个物品，价值随机在5到100之间
max_weight_pct = 0.4
fitness1 = mlrose.Knapsack(weights, values, max_weight_pct)
problem1 = mlrose.DiscreteOpt(length=len(weights), fitness_fn=fitness1, maximize=True)
problem1.set_mimic_fast_mode(True)

# 问题1：FourPeaks，强调遗传算法的优点
import numpy as np
import mlrose_hiive
from mlrose_hiive import DiscreteOpt, random_hill_climb, simulated_annealing, genetic_alg, mimic
from mlrose_hiive.fitness import FourPeaks, OneMax, FlipFlop

# 问题2：TSP，强调退火算法的优点

coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]

# 定义问题
fitness2 = mlrose.TravellingSales(coords=coords_list)
problem2 = mlrose.TSPOpt(length=len(coords_list), fitness_fn=fitness2, maximize=False)

problem2.set_mimic_fast_mode(True)
# 为模拟退火定义调度（温度函数）
schedule = mlrose_hiive.GeomDecay(init_temp=10, decay=0.95, min_temp=0.001)

fitness3 = FlipFlop()
problem3 = DiscreteOpt(length=200, fitness_fn=fitness3)
problem3.set_mimic_fast_mode(True)
problems = [problem1,problem2,problem3]

algorithms = ["Random Hill Climb", "Simulated Annealing", "Genetic Algorithm", "MIMIC"]
problem_names = ["Knapsack","TSP", "FlipFlop"]


def apply_algorithms(problem):
    # 随机爬山
    _, best_fitness_rhc, _ = random_hill_climb(problem, restarts=10, max_attempts=50, max_iters=50, random_state=42)
    # 模拟退火
    _, best_fitness_sa, _ = simulated_annealing(problem, schedule=mlrose_hiive.GeomDecay(), max_attempts=50, max_iters=50, random_state=42)
    # 遗传算法
    _, best_fitness_ga, _ = genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=50, max_iters=50, random_state=42)
    # MIMIC
    _, best_fitness_mimic, _ = mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=50, max_iters=50, random_state=42)
    
    
    return best_fitness_rhc, best_fitness_sa, best_fitness_ga, best_fitness_mimic,



results = [apply_algorithms(p) for p in problems]


for i, r in enumerate(results):
    print(f"Problem: {problem_names[i]}")
    for j, alg in enumerate(algorithms):
        print(f"{alg}: {r[j]}")
    print("\n")

def apply_algorithms(problem):
    # 注意curve=True的设置
    best_state_rhc, best_fitness_rhc, fitness_curve_rhc = random_hill_climb(problem, restarts=100, max_attempts=50, max_iters=50, random_state=42, curve=True)
    best_state_sa, best_fitness_sa, fitness_curve_sa = simulated_annealing(problem, schedule=schedule, max_attempts=50, max_iters=50, random_state=42, curve=True)
    best_state_ga, best_fitness_ga, fitness_curve_ga = genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=50, max_iters=50, random_state=42, curve=True)
    best_state_mimic, best_fitness_mimic, fitness_curve_mimic = mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=50, max_iters=50, random_state=42, curve=True)
    
    return fitness_curve_rhc, fitness_curve_sa, fitness_curve_ga, fitness_curve_mimic

results = [apply_algorithms(p) for p in problems]

for i, curves in enumerate(results):
    plt.figure(figsize=(12, 8))
    
    for curve, alg in zip(curves, algorithms):
        # cv = []
        # for k in range(len(curve)):
        #     cv.append([k,curve[k][0]])
        print(alg)
        print(curve)
   
        plt.plot(range(len(curve)),curve[:,1], label=alg)

    plt.title(f"Wall Clock Time/ Iters {problem_names[i]}")
    plt.xlabel("Iterations")
    plt.ylabel("WCT")
    plt.legend()
    plt.grid(True)
    plt.show()
# for i, curves in enumerate(results):
#     plt.figure(figsize=(12, 8))
    
#     for curve, alg in zip(curves, algorithms):
#         plt.plot(range(len(curve)),curve[:,1], label=alg)

#     plt.title(f"Fitness Curve for {problem_names[i]}")
#     plt.xlabel("Iterations")
#     plt.ylabel("Fitness")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

