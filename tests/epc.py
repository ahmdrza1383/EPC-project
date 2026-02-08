import numpy as np
import math
import time

class EPC_Optimizer:
    def __init__(self, objective_func, dim, bounds, population_size=30, max_iter=100):
        """
        Emperor Penguins Colony (EPC) Algorithm Implementation.
        Targeted for Co-design project simulation.
        """
        self.func = objective_func
        self.dim = dim
        self.lb = bounds[0]
        self.ub = bounds[1]
        self.pop_size = population_size
        self.max_iter = max_iter

        # --- Algorithm Parameters (Based on PDF) ---
        self.M = 0.05        # Initial movement range (Exploration)
        self.mu = 0.05       # Heat attenuation coefficient
        self.a = 1.0        # Spiral parameter a
        self.b = 0.5        # Spiral parameter b

        # Initialization
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.fitness = np.zeros(self.pop_size)

        # Global Best Record
        self.best_position = np.zeros(self.dim)
        self.best_score = float('inf')

    def check_bounds(self, position):
        """Clamps the solution within the defined search space."""
        return np.clip(position, self.lb, self.ub)

    def calculate_spiral_move(self, x_i, y_i, x_best, y_best, Q):
        """
        Calculates the new 2D position based on logarithmic spiral equations.
        Ref: PDF numerical example logic matches atan2(y, x).
        """
        # 1. Calculate Angles
        # Note: We use atan2(y, x) which is standard math notation
        theta_i = math.atan2(y_i, x_i)
        theta_j = math.atan2(y_best, x_best)

        # 2. Calculate Spiral Factor S
        # Formula: S = (1 - Q) * e^(b * theta_j) + Q * e^(b * theta_i)
        try:
            term1 = (1.0 - Q) * math.exp(self.b * theta_j)
            term2 = Q * math.exp(self.b * theta_i)
            S = term1 + term2
        except OverflowError:
            S = 1.0  # Safe fallback

        # Avoid math domain error for log
        if S <= 0: S = 1e-6

        # 3. Calculate New Angle and Radius
        theta_k = (1.0 / self.b) * math.log(S)
        r_k = self.a * S

        # 4. Convert back to Cartesian coordinates
        new_x = r_k * math.cos(theta_k)
        new_y = r_k * math.sin(theta_k)

        return new_x, new_y

    def run(self):
        print(f"{'='*30} EPC START {'='*30}")
        print(f"Population: {self.pop_size} | Dimensions: {self.dim} | Max Iterations: {self.max_iter}")
        print("-" * 75)

        start_time = time.time()

        for t in range(self.max_iter):
            # ---------------------------------------------------
            # Phase 1: Evaluation & Update Best
            # ---------------------------------------------------
            for i in range(self.pop_size):
                # Calculate cost
                cost = self.func(self.population[i])
                self.fitness[i] = cost

                # Update Global Best
                if cost < self.best_score:
                    self.best_score = cost
                    self.best_position = self.population[i].copy()

            # ---------------------------------------------------
            # Logging: Print status every iteration
            # ---------------------------------------------------
            print(f"Iter {t+1:3d}/{self.max_iter} | Best Cost: {self.best_score:.6e} | "
                  f"mu: {self.mu:.4f} | M: {self.M:.4f}")

            # ---------------------------------------------------
            # Phase 2: Movement (All-Pairs Strategy)
            # ---------------------------------------------------
            new_population = np.zeros_like(self.population)

            for i in range(self.pop_size):
                # Elitism: Keep the best penguin stable
                if np.array_equal(self.population[i], self.best_position):
                    new_population[i] = self.population[i]
                    continue

                # A) Distance and Heat Calculation
                dist = np.linalg.norm(self.population[i] - self.best_position)
                # Q = e^(-mu * dist)
                try:
                    Q = math.exp(-self.mu * dist)
                except OverflowError:
                    Q = 0.0 # If distance is huge, heat is zero
                
                Q = np.clip(Q, 0.0001, 1.0) # Clamp Q to valid range

                # B) Spiral Movement (All-Pairs Interaction)
                accumulated_moves = np.zeros(self.dim)
                move_counts = np.zeros(self.dim)

                current_p = self.population[i]

                # Iterate over all unique pairs of dimensions
                for d1 in range(self.dim):
                    for d2 in range(d1 + 1, self.dim):
                        # Coordinates for the current pair
                        curr_x, curr_y = current_p[d1], current_p[d2]
                        best_x, best_y = self.best_position[d1], self.best_position[d2]

                        # Calculate spiral move
                        nx, ny = self.calculate_spiral_move(curr_x, curr_y, best_x, best_y, Q)

                        # Accumulate proposed moves
                        accumulated_moves[d1] += nx
                        move_counts[d1] += 1
                        accumulated_moves[d2] += ny
                        move_counts[d2] += 1

                # C) Average and Apply Moves
                updated_penguin = np.zeros(self.dim)
                for d in range(self.dim):
                    if move_counts[d] > 0:
                        updated_penguin[d] = accumulated_moves[d] / move_counts[d]
                    else:
                        updated_penguin[d] = current_p[d]

                # D) Add Randomness (Exploration)
                random_step = self.M * np.random.uniform(-1, 1, self.dim)
                updated_penguin += random_step

                # E) Boundary Check
                new_population[i] = self.check_bounds(updated_penguin)

            # Apply new population
            self.population = new_population

            # Update control parameters (Cooling mechanism)
            if t > 0:
                self.M *= 0.99
                self.mu *= 0.99

        total_time = time.time() - start_time
        print("-" * 75)
        print(f"Optimization Finished.")
        print(f"Final Best Cost: {self.best_score:.10f}")
        print(f"Total Execution Time: {total_time:.4f} sec")
        print(f"{'='*30} EPC END {'='*30}")
        
        return self.best_score

# -------------------------------------------------------
# Benchmark Functions
# -------------------------------------------------------
def sphere_function(x):
    """
    [cite_start]Sphere Function [cite: 140]
    Global Minimum: 0 at (0,0,...,0)
    """
    return np.sum(x**2)

def rosenbrock_function(x):
    """
    [cite_start]Rosenbrock Function [cite: 142]
    Global Minimum: 0 at (1,1,...,1)
    """
    sum_val = 0
    # Loop from 0 to D-2 (corresponds to i=1 to D-1 in math notation)
    for i in range(len(x) - 1):
        term1 = 100 * (x[i+1] - x[i]**2)**2
        term2 = (x[i] - 1)**2
        sum_val += term1 + term2
    return sum_val

# -------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    # --- SELECT FUNCTION HERE ---
    # Options: "sphere" or "rosenbrock"
    SELECTED_FUNC = "sphere" 
    
    # Simulation Parameters
    DIMENSIONS = 100     # Try 4, 10, or 30
    POPULATION = 20
    ITERATIONS = 100
    
    if SELECTED_FUNC == "sphere":
        objective_func = sphere_function
        SEARCH_SPACE = (-5.12, 5.12) # Standard bounds for Sphere
        print(">>> Running SPHERE Benchmark")
        
    elif SELECTED_FUNC == "rosenbrock":
        objective_func = rosenbrock_function
        SEARCH_SPACE = (-5, 10)   # Rosenbrock is sensitive, smaller bounds are safer
        print(">>> Running ROSENBROCK Benchmark")

    # Run Optimizer
    optimizer = EPC_Optimizer(objective_func, DIMENSIONS, SEARCH_SPACE, POPULATION, ITERATIONS)
    optimizer.run()