import numpy as np
import matplotlib.pyplot as plt
import math
import time
import copy


# =============================================================================
# Class: EPC_Optimizer
# Description: Full implementation of Emperor Penguins Colony Algorithm
#              Supports both Hardware-Friendly (Disjoint) and Theoretical (All-Pairs) strategies.
# =============================================================================
class EPC_Optimizer:
    def __init__(
        self,
        objective_func,
        dim,
        bounds,
        population_size=30,
        max_iter=100,
        initial_pop=None,
        mode="min",
    ):
        """
        Initialize the EPC Algorithm with all necessary parameters.

        Args:
            objective_func: The function to minimize.
            dim: Number of dimensions (D).
            bounds: Tuple (min, max) for search space limits.
            population_size: Number of penguins (N).
            max_iter: Maximum number of iterations (T_max).
            initial_pop: Optional shared population for fair comparison.
        """
        self.func = objective_func
        self.dim = dim
        self.lb = bounds[0]  # Lower Bound
        self.ub = bounds[1]  # Upper Bound
        self.pop_size = population_size
        self.max_iter = max_iter

        # --- Algorithm Parameters (From PDF Pages 6 & 7) ---
        self.M = 2.0  # Initial movement range (Exploration) [cite: 134]
        self.mu = 0.5  # Heat attenuation coefficient [cite: 113]
        self.a = 1.0  # Spiral parameter a [cite: 118]
        self.b = 0.5  # Spiral parameter b [cite: 118]

        # --- Initialization ---
        # Use provided population if available (for comparing strategies), else random.
        if initial_pop is not None:
            self.population = copy.deepcopy(initial_pop)
        else:
            # Random initialization [cite: 76]
            self.population = np.random.uniform(
                self.lb, self.ub, (self.pop_size, self.dim)
            )

        self.fitness = np.zeros(self.pop_size)

        self.mode = mode

        # Global Best (The solution we are looking for)
        self.best_penguin_pos = np.zeros(self.dim)
        self.best_penguin_score = float("inf") if mode == "min" else float("-inf")

        # History for convergence plot
        self.convergence_curve = []

    def check_bounds(self, position):
        """
        Clamps the position to ensure it stays within [LB, UB].
        Represents saturation hardware logic. [cite: 136]
        """
        return np.clip(position, self.lb, self.ub)

    def calculate_spiral_point(self, p_curr, p_best, Q):
        """
        Calculates the new 2D position based on Logarithmic Spiral.
        Math Reference: PDF Page 7 & 11

        Args:
            p_curr: tuple(x, y) of current penguin
            p_best: tuple(x, y) of best penguin
            Q: Heat absorption factor

        Returns:
            new_x, new_y: Updated coordinates
        """
        x_i, y_i = p_curr
        x_j, y_j = p_best

        # 1. Calculate Angles (Theta)
        theta_i = math.atan2(y_i, x_i)
        theta_j = math.atan2(y_j, x_j)

        # 2. Calculate Spiral Factor S [cite: 123, 183]
        # Formula: S = (1 - Q) * e^(b * theta_j) + Q * e^(b * theta_i)
        term1 = (1 - Q) * math.exp(self.b * theta_j)
        term2 = Q * math.exp(self.b * theta_i)
        S = term1 + term2

        # 3. Calculate New Angle (Theta_k) [cite: 123, 185]
        # Formula: theta_k = (1/b) * ln(S)
        try:
            # Guard against math domain error if S <= 0 (rare)
            if S <= 0:
                S = 1e-6
            theta_k = (1.0 / self.b) * math.log(S)
        except ValueError:
            theta_k = theta_j

        # 4. Calculate Radius (r_k) [cite: 130]
        # Formula: r = a * e^(b * theta_k) which simplifies to r = a * S
        r_k = self.a * S

        # 5. Convert back to Cartesian [cite: 132]
        new_x = r_k * math.cos(theta_k)
        new_y = r_k * math.sin(theta_k)

        return new_x, new_y

    def update_parameters(self, iteration):
        """
        Updates dynamic parameters M and mu over time.
        This simulates cooling and convergence.
        """
        if iteration > 0:
            self.M *= 0.99  # Decrease mutation range [cite: 134]
            self.mu *= 0.99  # Decrease heat attenuation [cite: 112]

    def run(self, strategy="disjoint"):
        """
        Main Execution Loop.

        Args:
            strategy (str):
                'disjoint': Optimized for hardware (Pairwise Decomposition).
                'all_pairs': Full theoretical interaction (as per Doc Page 10).
        """
        print(f"Running EPC with strategy: {strategy.upper()}")
        start_time = time.time()

        for t in range(self.max_iter):
            # -------------------------------------------------------
            # Phase 1: Evaluation & Identifying Best Penguin
            # -------------------------------------------------------
            if self.mode == "min":
                current_best_val = float("inf")
            else:
                current_best_val = float("-inf")

            current_best_idx = -1

            for i in range(self.pop_size):
                # محاسبه هزینه
                cost = self.func(self.population[i])
                self.fitness[i] = cost

                # --- الف) پیدا کردن بهترین در جمعیت فعلی (برای Elitism) ---
                is_local_better = False
                if self.mode == "min":
                    if cost < current_best_val:
                        is_local_better = True
                else:  # max
                    if cost > current_best_val:
                        is_local_better = True

                if is_local_better:
                    current_best_val = cost
                    current_best_idx = i

                # --- ب) پیدا کردن بهترین کل تاریخچه (Global Best) ---
                # (این همان کدی است که خودت نوشتی)
                is_global_better = False
                if self.mode == "min":
                    if cost < self.best_penguin_score:
                        is_global_better = True
                else:  # max
                    if cost > self.best_penguin_score:
                        is_global_better = True

                if is_global_better:
                    self.best_penguin_score = cost
                    self.best_penguin_pos = self.population[i].copy()
            # Save history
            self.convergence_curve.append(self.best_penguin_score)

            # -------------------------------------------------------
            # Phase 2: Movement Strategy
            # -------------------------------------------------------
            new_population = np.zeros_like(self.population)

            for i in range(self.pop_size):
                # [Elitism] The best penguin of current generation stays put
                # We use index comparison to avoid floating point equality issues
                if i == current_best_idx:
                    new_population[i] = self.population[i]
                    continue

                # 1. Calculate Euclidean Distance
                distance = np.linalg.norm(self.best_penguin_pos - self.population[i])

                # 2. Calculate Heat/Attraction Q
                # Q = e^(-mu * distance)
                Q = math.exp(-self.mu * distance)
                Q = max(0.0001, min(Q, 1.0))  # Clamp Q

                current_penguin = self.population[i]
                updated_penguin = np.zeros(self.dim)

                # ======================================================
                # STRATEGY A: DISJOINT PAIRS (Hardware Optimized)
                # Matches the logic needed for parallel SystemC modules
                # ======================================================
                if strategy == "disjoint":
                    for d in range(0, self.dim, 2):
                        if d + 1 < self.dim:
                            # Pair (d, d+1) -> 2D Plane
                            curr_pair = (current_penguin[d], current_penguin[d + 1])
                            best_pair = (
                                self.best_penguin_pos[d],
                                self.best_penguin_pos[d + 1],
                            )

                            nx, ny = self.calculate_spiral_point(
                                curr_pair, best_pair, Q
                            )
                            updated_penguin[d], updated_penguin[d + 1] = nx, ny
                        else:
                            # Odd dimension handling (Linear move)
                            diff = self.best_penguin_pos[d] - current_penguin[d]
                            updated_penguin[d] = current_penguin[d] + diff * Q

                # ======================================================
                # STRATEGY B: ALL PAIRS (Theoretical / Doc Page 10)
                # Every dimension interacts with every other dimension.
                # Complexity: O(D^2) - Very heavy for hardware
                # ======================================================
                elif strategy == "all_pairs":
                    # We need accumulators because a single dimension (e.g., x1)
                    # is part of multiple pairs: (x1,x2), (x1,x3), (x1,x4)...
                    move_sum = np.zeros(self.dim)
                    move_count = np.zeros(self.dim)

                    # Use a temp copy so we don't read updated values mid-loop
                    temp_p = current_penguin.copy()

                    # Iterate over ALL unique combinations
                    for d1 in range(self.dim):
                        for d2 in range(d1 + 1, self.dim):
                            curr_pair = (temp_p[d1], temp_p[d2])
                            best_pair = (
                                self.best_penguin_pos[d1],
                                self.best_penguin_pos[d2],
                            )

                            nx, ny = self.calculate_spiral_point(
                                curr_pair, best_pair, Q
                            )

                            # Accumulate the proposed new positions
                            move_sum[d1] += nx
                            move_count[d1] += 1

                            move_sum[d2] += ny
                            move_count[d2] += 1

                    # Average the moves to get final position
                    for d in range(self.dim):
                        if move_count[d] > 0:
                            updated_penguin[d] = move_sum[d] / move_count[d]
                        else:
                            updated_penguin[d] = temp_p[d]

                # 3. Add Randomness (Exploration) [cite: 134]
                # x_new += M * u
                random_step = self.M * (np.random.uniform(-1, 1, self.dim))
                updated_penguin += random_step

                # 4. Check Bounds [cite: 135]
                new_population[i] = self.check_bounds(updated_penguin)

            # Update population for next generation
            self.population = new_population

            # Decay parameters
            self.update_parameters(t)

        exec_time = time.time() - start_time
        return self.best_penguin_score, self.convergence_curve, exec_time


# =============================================================================
# Benchmark Functions [cite: 139, 141]
# =============================================================================


def sphere(x):
    """Sphere Function: f(x) = sum(x^2). Min = 0 at (0,0...)"""
    return np.sum(x**2)


def rosenbrock(x):
    """Rosenbrock Function. Min = 0 at (1,1...)"""
    sum_val = 0
    for i in range(len(x) - 1):
        sum_val += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2
    return sum_val


# =============================================================================
# Main Execution & Verification
# =============================================================================

if __name__ == "__main__":
    # Test Settings
    DIM = 4  # Dimensions (e.g., 4 as per Page 9)
    POP = 30  # Population Size
    ITERS = 100  # Total Iterations
    BOUNDS = (-10, 10)  # Search Space

    # 1. Generate a SHARED Initial Population
    # Ensures both strategies start from exact same random points
    shared_population = np.random.uniform(BOUNDS[0], BOUNDS[1], (POP, DIM))

    print("--- Comparing EPC Strategies for Hardware Co-Design ---")

    # -------------------------------------------------------
    # Test 1: Disjoint Pairs (Hardware Friendly)
    # -------------------------------------------------------
    # This logic matches what we will implement in SystemC
    epc_disjoint = EPC_Optimizer(
        sphere, DIM, BOUNDS, POP, ITERS, initial_pop=shared_population
    )
    score_A, curve_A, time_A = epc_disjoint.run(strategy="disjoint")

    # -------------------------------------------------------
    # Test 2: All Pairs (Theoretical / Documentation)
    # -------------------------------------------------------
    # This logic matches the full combinatorial pairs in the PDF
    epc_all = EPC_Optimizer(
        sphere, DIM, BOUNDS, POP, ITERS, initial_pop=shared_population
    )
    score_B, curve_B, time_B = epc_all.run(strategy="all_pairs")

    # -------------------------------------------------------
    # Reporting Results
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"{'Metric':<20} | {'Disjoint (HW Opt)':<22} | {'All-Pairs (Theory)':<20}")
    print("-" * 60)
    print(f"{'Final Cost':<20} | {score_A:.6e}{'':<10} | {score_B:.6e}")
    print(f"{'Execution Time':<20} | {time_A:.6f} sec{'':<10} | {time_B:.6f} sec")
    print(f"{'Complexity':<20} | O(D) - Linear{'':<9} | O(D^2) - Quadratic")
    print("=" * 60)

    # -------------------------------------------------------
    # Visualization
    # -------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(curve_A, label="Disjoint (Proposed for SystemC)", linewidth=2)
    plt.plot(curve_B, label="All-Pairs (Doc Reference)", linewidth=2, linestyle="--")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Cost (Log Scale)")
    plt.title(f"Comparison of EPC Strategies (Dim={DIM})")
    plt.legend()
    plt.grid(True, which="both", alpha=0.5)

    # Save plot for report
    plt.savefig("epc_strategy_comparison.png")
    print("\n[Info] Comparison plot saved as 'epc_strategy_comparison.png'")

    plt.show()
