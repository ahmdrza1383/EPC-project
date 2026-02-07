import numpy as np
import matplotlib.pyplot as plt
import math
import csv

# =============================================================================
# Class: EPC_Optimizer
# Description: Implements the Emperor Penguins Colony algorithm.
#              This serves as the Golden Model for hardware verification.
# =============================================================================
class EPC_Optimizer:
    def __init__(self, objective_func, dim, bounds, population_size=30, max_iter=100):
        """
        Initialize the EPC Optimizer parameters.
        
        Args:
            objective_func (function): The cost function to minimize (e.g., Sphere).
            dim (int): Number of dimensions (variables).
            bounds (tuple): A tuple (lower_bound, upper_bound) for search space.
            population_size (int): Number of penguins in the colony.
            max_iter (int): Maximum number of iterations (epochs).
        """
        self.func = objective_func
        self.dim = dim
        self.lb = bounds[0] # Lower Bound
        self.ub = bounds[1] # Upper Bound
        self.pop_size = population_size
        self.max_iter = max_iter
        
        # --- Algorithm Parameters (Based on the PDF) ---
        self.M = 2.0        # Initial movement randomness parameter (page 7)
        self.mu = 0.5       # Heat attenuation coefficient (page 6)
        self.a = 1.0        # Spiral shape parameter a (page 6)
        self.b = 0.5        # Spiral shape parameter b (page 6)
        
        # --- State Variables ---
        # Initialize population with random values within bounds
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.fitness = np.zeros(self.pop_size)
        self.best_penguin_pos = np.zeros(self.dim)
        self.best_penguin_score = float('inf')
        self.best_penguin_idx = -1  
        
        # History for plotting and analysis
        self.convergence_curve = []

    def check_bounds(self, position):
        """
        Clamp the position vector to ensure it stays within the search space.
        Equivalent to the hardware saturation logic.
        """
        return np.clip(position, self.lb, self.ub)

    def calculate_pairwise_spiral(self, p_curr, p_best, Q):
        """
        Perform the spiral movement calculation for a pair of dimensions.
        This function represents the logic shown on page 11 of the PDF.
        
        Args:
            p_curr (tuple): (x, y) coordinates of the current penguin.
            p_best (tuple): (x, y) coordinates of the best penguin.
            Q (float): Heat absorption factor.
            
        Returns:
            tuple: New (x, y) coordinates after spiral movement.
        """
        x_i, y_i = p_curr
        x_j, y_j = p_best # Best penguin is the target (j)

        # Calculate angles (theta)
        # Note: atan2 handles the quadrants correctly
        theta_i = math.atan2(y_i, x_i) 
        theta_j = math.atan2(y_j, x_j)

        # Calculate the spiral scaling factor S (Page 11, Eq: 183)
        # S = (1 - Q) * e^(b * theta_j) + Q * e^(b * theta_i)
        # Note: We must ensure the exponential doesn't overflow
        term1 = (1 - Q) * math.exp(self.b * theta_j)
        term2 = Q * math.exp(self.b * theta_i)
        S = term1 + term2

        # Calculate new angle and radius (Page 11, Eq: 185, 186)
        # theta_k = (1/b) * ln(S)
        try:
            theta_k = (1.0 / self.b) * math.log(S)
        except ValueError:
            # Handle math domain errors if S <= 0 (rare but possible)
            theta_k = theta_j 

        r_k = self.a * S

        # Convert back to Cartesian coordinates
        x_new = r_k * math.cos(theta_k)
        y_new = r_k * math.sin(theta_k)

        return x_new, y_new

    def update_parameters(self, iteration):
        """
        Update dynamic parameters M and mu to reduce exploration over time.
        This simulates the 'cooling' process.
        """
        if iteration > 0:
            self.M *= 0.99   # Decay randomness (Page 7)
            self.mu *= 0.99  # Decay heat attenuation (Page 6)

    def run(self):
        """
        Main execution loop of the EPC algorithm.
        Executes the optimization process for 'max_iter' generations.
        """
        print(f"Starting EPC Optimization on {self.dim}-D Space...")
        
        for t in range(self.max_iter):
            # -------------------------------------------------------
            # Phase 1: Evaluation & Identifying the Leader
            # -------------------------------------------------------
            
            # We need to track the best penguin of the *current* generation
            # to prevent it from moving (Elitism).
            current_best_val = float('inf')
            current_best_idx = -1

            for i in range(self.pop_size):
                # 1. Calculate cost/fitness
                current_cost = self.func(self.population[i])
                self.fitness[i] = current_cost
                
                # 2. Identify the best in current population (Local Best)
                # We save the index to use it later in the movement phase
                if current_cost < current_best_val:
                    current_best_val = current_cost
                    current_best_idx = i

                # 3. Update Global Best (Historical Best)
                # If this penguin is better than the all-time best, update the record
                if current_cost < self.best_penguin_score:
                    self.best_penguin_score = current_cost
                    self.best_penguin_pos = self.population[i].copy()

            # Record history for the convergence plot
            self.convergence_curve.append(self.best_penguin_score)

            # -------------------------------------------------------
            # Phase 2: Movement & Update Strategy
            # -------------------------------------------------------
            new_population = np.zeros_like(self.population)
            
            for i in range(self.pop_size):
                # [Elitism Strategy]
                # If this is the best penguin of the current generation, it stays put.
                # Comparing indices (i == current_best_idx) is safer than comparing arrays.
                if i == current_best_idx:
                    new_population[i] = self.population[i]
                    continue

                # 1. Calculate Euclidean Distance to the Global Best (Page 6, Eq: 119)
                distance = np.linalg.norm(self.best_penguin_pos - self.population[i])
                
                # 2. Calculate Heat/Attraction Q (Page 6, Eq: 111)
                # Q = e^(-mu * distance) -> Decays as distance increases
                Q = math.exp(-self.mu * distance)
                Q = max(0.0001, min(Q, 1.0)) # Clamp Q between 0 and 1

                # 3. Spiral Update (Pairwise Decomposition Strategy)
                # We process dimensions in pairs (0,1), (2,3) to apply 2D spiral math.
                current_penguin = self.population[i]
                updated_penguin = np.zeros(self.dim)
                
                for d in range(0, self.dim, 2):
                    if d + 1 < self.dim:
                        # Case A: Pair exists (e.g., dimensions x and y)
                        curr_pair = (current_penguin[d], current_penguin[d+1])
                        best_pair = (self.best_penguin_pos[d], self.best_penguin_pos[d+1])
                        
                        # Apply spiral math on this 2D plane
                        new_x, new_y = self.calculate_pairwise_spiral(curr_pair, best_pair, Q)
                        
                        updated_penguin[d] = new_x
                        updated_penguin[d+1] = new_y
                    else:
                        # Case B: Odd dimension remaining (Linear movement)
                        # Move directly towards the best position scaled by Q
                        updated_penguin[d] = current_penguin[d] + (self.best_penguin_pos[d] - current_penguin[d]) * Q

                # 4. Add Randomness (Exploration) (Page 7, Eq: 134)
                # Adds a stochastic component to prevent getting stuck in local optima
                random_step = self.M * (np.random.uniform(-1, 1, self.dim))
                updated_penguin += random_step
                
                # 5. Boundary Check (Saturation)
                new_population[i] = self.check_bounds(updated_penguin)

            # -------------------------------------------------------
            # Phase 3: Finalize Iteration
            # -------------------------------------------------------
            # Synchronous update: Replace old population with the new one
            self.population = new_population
            
            # Cool down the parameters (Temperature & Randomness decay)
            self.update_parameters(t)

            # Log progress
            if (t + 1) % 10 == 0:
                print(f"Iteration {t+1:03d}: Best Cost = {self.best_penguin_score:.6e}")

        return self.best_penguin_pos, self.best_penguin_score
# =============================================================================
# Helper Functions: Benchmarks and Visualization
# =============================================================================

def sphere_function(x):
    """
    Standard Sphere benchmark function.
    Global minimum is 0 at (0,0,...,0).
    Formula: f(x) = sum(x_i^2)
    """
    return np.sum(x**2)

def rosenbrock_function(x):
    """
    Standard Rosenbrock benchmark function.
    Global minimum is 0 at (1,1,...,1).
    """
    sum_val = 0
    for i in range(len(x)-1):
        sum_val += 100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
    return sum_val

def save_results_to_csv(history, filename="epc_results.csv"):
    """Saves the convergence data to a CSV file for SystemC comparison."""
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Iteration", "Best_Cost"])
        for idx, val in enumerate(history):
            writer.writerow([idx + 1, val])
    print(f"\n[Info] Results saved to {filename}")

# =============================================================================
# Main Execution Block
# =============================================================================
if __name__ == "__main__":
    # Project Settings
    DIMENSION = 4           # As per the example in the PDF
    POPULATION = 30         # Number of penguins
    ITERATIONS = 100        # Number of cycles
    BOUNDS = (-10, 10)      # Search space
    
    # 1. Run on Sphere Function
    print("--- Running EPC on Sphere Function ---")
    optimizer_sphere = EPC_Optimizer(sphere_function, DIMENSION, BOUNDS, POPULATION, ITERATIONS)
    best_pos, best_score = optimizer_sphere.run()
    
    print(f"\nFinal Best Position: {np.round(best_pos, 4)}")
    print(f"Final Best Cost: {best_score:.6e}")
    
    # Save results for comparison
    save_results_to_csv(optimizer_sphere.convergence_curve, "epc_sphere_golden.csv")

    # 2. Run on Rosenbrock Function
    print("\n--- Running EPC on Rosenbrock Function ---")
    optimizer_rosen = EPC_Optimizer(rosenbrock_function, DIMENSION, BOUNDS, POPULATION, ITERATIONS)
    best_pos_r, best_score_r = optimizer_rosen.run()
    
    print(f"\nFinal Best Position: {np.round(best_pos_r, 4)}")
    print(f"Final Best Cost: {best_score_r:.6e}")

    # 3. Plotting Results
    plt.figure(figsize=(10, 6))
    plt.plot(optimizer_sphere.convergence_curve, label='Sphere', linewidth=2)
    plt.plot(optimizer_rosen.convergence_curve, label='Rosenbrock', linewidth=2, linestyle='--')
    plt.yscale('log')  # Logarithmic scale to see convergence details clearly
    plt.xlabel('Iteration')
    plt.ylabel('Cost (Best Fitness)')
    plt.title('EPC Algorithm Convergence (Python Golden Model)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig('epc_convergence.png')
    print("\n[Info] Plot saved as 'epc_convergence.png'")
    plt.show()