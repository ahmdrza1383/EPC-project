import numpy as np
import math
import time

class EPC_Optimizer_Final:
    def __init__(self, objective_func, dim, bounds, population_size=30, max_iter=100, 
                 strategy='all_pairs', optimization_mode='min'):
        self.func = objective_func
        self.dim = dim
        self.lb = bounds[0]
        self.ub = bounds[1]
        self.pop_size = population_size
        self.max_iter = max_iter
        self.strategy = strategy
        self.mode = optimization_mode.lower()

        self.M = 0.5
        self.mu = 0.05
        self.a = 1.0
        self.b = 0.5

        if self.strategy == 'random':
            self.n_neighbors = max(2, int(math.sqrt(self.dim)))
        else:
            self.n_neighbors = self.dim - 1

        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.fitness = np.zeros(self.pop_size)

        self.global_best_pos = np.zeros(self.dim)
        if self.mode == 'min':
            self.global_best_score = float('inf')
        else:
            self.global_best_score = float('-inf')

    def check_bounds(self, position):
        return np.clip(position, self.lb, self.ub)

    def is_better(self, new_score, current_score):
        if self.mode == 'min':
            return new_score < current_score
        else:
            return new_score > current_score

    def calculate_spiral_move(self, x_i, y_i, x_best, y_best, Q):
        theta_i = math.atan2(y_i, x_i)
        theta_j = math.atan2(y_best, x_best)

        try:
            term1 = (1.0 - Q) * math.exp(self.b * theta_j)
            term2 = Q * math.exp(self.b * theta_i)
            S = term1 + term2
        except OverflowError:
            S = 1.0

        if S <= 0: S = 1e-6

        theta_k = (1.0 / self.b) * math.log(S)
        r_k = self.a * S

        new_x = r_k * math.cos(theta_k)
        new_y = r_k * math.sin(theta_k)

        return new_x, new_y

    def run(self):
        print(f"{'='*30} EPC FINAL {'='*30}")
        print(f"Mode: {self.mode.upper()} | Strategy: {self.strategy.upper()}")
        print(f"Pop={self.pop_size} | Dim={self.dim} | Iter={self.max_iter}")
        print("-" * 75)

        start_time = time.time()

        for t in range(self.max_iter):
            iter_best_score = float('inf') if self.mode == 'min' else float('-inf')
            iter_best_idx = -1

            for i in range(self.pop_size):
                cost = self.func(self.population[i])
                self.fitness[i] = cost

                if self.is_better(cost, iter_best_score):
                    iter_best_score = cost
                    iter_best_idx = i
                
                if self.is_better(cost, self.global_best_score):
                    self.global_best_score = cost
                    self.global_best_pos = self.population[i].copy()

            iter_best_position = self.population[iter_best_idx].copy()

            print(f"Iter {t+1:3d}/{self.max_iter} | "
                  f"Global: {self.global_best_score:.6e} | "
                  f"IterBest: {iter_best_score:.6e} | M: {self.M:.4f}")

            new_population = np.zeros_like(self.population)

            for i in range(self.pop_size):
                temp_position = np.zeros(self.dim)

                if i == iter_best_idx:
                    temp_position = self.population[i].copy()
                
                else:
                    dist = np.linalg.norm(self.population[i] - iter_best_position)
                    try:
                        Q = math.exp(-self.mu * dist)
                    except OverflowError:
                        Q = 0.0
                    Q = np.clip(Q, 0.0001, 1.0)

                    accumulated_moves = np.zeros(self.dim)
                    move_counts = np.zeros(self.dim)
                    current_p = self.population[i]

                    for d1 in range(self.dim):
                        if self.strategy == 'random':
                            candidates = np.delete(np.arange(self.dim), d1)
                            n_pick = min(len(candidates), self.n_neighbors)
                            partners = np.random.choice(candidates, n_pick, replace=False)
                        else:
                            partners = range(d1 + 1, self.dim)

                        for d2 in partners:
                            curr_x, curr_y = current_p[d1], current_p[d2]
                            best_x, best_y = iter_best_position[d1], iter_best_position[d2]

                            nx, ny = self.calculate_spiral_move(curr_x, curr_y, best_x, best_y, Q)

                            accumulated_moves[d1] += nx
                            move_counts[d1] += 1
                            
                            if self.strategy == 'all_pairs':
                                accumulated_moves[d2] += ny
                                move_counts[d2] += 1

                    for d in range(self.dim):
                        if move_counts[d] > 0:
                            temp_position[d] = accumulated_moves[d] / move_counts[d]
                        else:
                            temp_position[d] = current_p[d]

                random_step = self.M * np.random.uniform(-1, 1, self.dim)
                final_position = temp_position + random_step

                new_population[i] = self.check_bounds(final_position)

            self.population = new_population

            if t > 0:
                self.M *= 0.99
                self.mu *= 0.99

        total_time = time.time() - start_time
        print("-" * 75)
        print(f"Finished. Global Best ({self.mode}): {self.global_best_score:.10f}")
        print(f"Execution Time: {total_time:.4f} sec")
        print(f"{'='*30} EPC END {'='*30}")
        return self.global_best_score

def sphere_function(x):
    return np.sum(x**2)

def rosenbrock_function(x):
    sum_val = 0
    for i in range(len(x) - 1):
        term1 = 100 * (x[i+1] - x[i]**2)**2
        term2 = (x[i] - 1)**2
        sum_val += term1 + term2
    return sum_val

if __name__ == "__main__":
    np.random.seed(50) 

    FUNC_NAME = "sphere" 
    
    if FUNC_NAME == "sphere":
        func = sphere_function
        bounds = (-10, 10)
    else:
        func = rosenbrock_function
        bounds = (-10, 10)

    optimizer = EPC_Optimizer_Final(
        objective_func=func,
        dim=100,
        bounds=bounds,
        population_size=20,
        max_iter=100,
        strategy='random',
        optimization_mode='min'
    )
    
    optimizer.run()