import itertools
import numpy as np
from scipy.optimize import minimize

def simulate_gaussian_performance(decimal_odds, num_simulations=100000, s=1.0, variability_factor=0.5):
    """
    Simulates races using Gaussian performance distributions for each horse.
    Calculates ERB% based on simulation results.
    """
    horses = list(decimal_odds.keys())
    q_values = np.array([1 / decimal_odds[horse] for horse in horses])

    # Normalize probabilities
    q_values = q_values / np.sum(q_values)

    # Assign means
    mu_values = -np.log(q_values)

    # Assign standard deviations with controlled variability
    sigma_values = s * (1 + variability_factor * (1 - q_values))

    # Initialize counters for finishing positions
    num_horses = len(horses)
    finish_counts = {horse: np.zeros(num_horses) for horse in horses}
    erb_counts = {horse: 0 for horse in horses}

    for _ in range(num_simulations):
        # Sample performance scores
        performance_scores = np.random.normal(loc=mu_values, scale=sigma_values)
        # Determine finishing order (lower score is better)
        finishing_order_indices = performance_scores.argsort()
        finishing_order = [horses[i] for i in finishing_order_indices]
        # Update finish counts
        for position, horse in enumerate(finishing_order):
            finish_counts[horse][position] += 1
            # Count number of rivals beaten
            erb_counts[horse] += num_horses - position - 1

    # Calculate place probabilities
    place_probabilities = {horse: finish_counts[horse] / num_simulations for horse in horses}

    # Calculate ERB%
    ERB_percentage = {}
    for horse in horses:
        avg_rivals_beaten = erb_counts[horse] / num_simulations
        ERB_percentage[horse] = (avg_rivals_beaten / (num_horses - 1)) * 100

    return ERB_percentage, place_probabilities

def backfit_odds_from_erb(target_erb, num_simulations=100000, s=1.0, variability_factor=0.5):
    """
    Backfits the win odds that produce the given target ERB% using Gaussian simulation.
    """
    horses = list(target_erb.keys())
    num_horses = len(horses)

    def loss(odds):
        # Convert odds to dictionary format for simulation
        decimal_odds = {horse: odd for horse, odd in zip(horses, odds)}
        simulated_erb, _ = simulate_gaussian_performance(decimal_odds, num_simulations, s, variability_factor)
        # Calculate mean squared error between target and simulated ERB%
        return sum((simulated_erb[horse] - target_erb[horse]) ** 2 for horse in horses)

    # Initial guess for odds: uniform odds
    initial_odds = np.full(num_horses, 10.0)
    bounds = [(1.01, 1000.0)] * num_horses  # Odds must be greater than 1.0

    # Optimize the odds to minimize the loss
    result = minimize(loss, initial_odds, bounds=bounds, method="L-BFGS-B")

    if not result.success:
        raise ValueError("Optimization failed: " + result.message)

    optimized_odds = result.x
    return {horse: odd for horse, odd in zip(horses, optimized_odds)}

if __name__ == "__main__":
    # Prompt user to input the target ERB%
    target_erb = {}
    num_horses = int(input("Enter the number of horses: "))
    print("Please enter the horse's name and their target ERB%.")
    for _ in range(num_horses):
        horse_name = input("Horse name: ")
        while True:
            try:
                erb_percentage = float(input(f"Target ERB% for {horse_name}: "))
                if erb_percentage < 0 or erb_percentage > 100:
                    print("ERB% must be between 0 and 100. Please try again.")
                    continue
                break
            except ValueError:
                print("Invalid input. Please enter a numerical value for the ERB%.")
        target_erb[horse_name] = erb_percentage

    # Backfit the odds from the given target ERB%
    optimized_odds = backfit_odds_from_erb(target_erb)

    # Output the results
    print("\nOptimized Win Odds Based on Target ERB%:")
    for horse, odd in optimized_odds.items():
        print(f"{horse}: Decimal Odds = {odd:.2f}")
