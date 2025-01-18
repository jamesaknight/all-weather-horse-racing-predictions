import itertools
import numpy as np
from scipy.optimize import bisect

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

if __name__ == "__main__":
    # Prompt user to input the odds
    decimal_odds_input = {}
    num_horses = int(input("Enter the number of horses: "))
    print("Please enter the horse's name and their decimal odds.")
    for _ in range(num_horses):
        horse_name = input("Horse name: ")
        while True:
            try:
                horse_odds = float(input(f"Decimal odds for {horse_name}: "))
                if horse_odds <= 0:
                    print("Odds must be a positive number. Please try again.")
                    continue
                break
            except ValueError:
                print("Invalid input. Please enter a numerical value for the odds.")
        decimal_odds_input[horse_name] = horse_odds

    # Simulate races using Gaussian performance distributions
    num_simulations = 100000  # Adjust as needed
    s = 1.0  # Scaling factor for standard deviations
    variability_factor = 0.5  # Adjust this value to control variability
    ERB_percentage, place_probabilities = simulate_gaussian_performance(
        decimal_odds_input, num_simulations, s, variability_factor)

    # Calculate place odds (inverse of the probability of finishing in top positions)
    place_odds = {}
    top_k = 3  # Adjust k as needed for 'place' definition
    for horse, probabilities in place_probabilities.items():
        # Probability of finishing in the top k positions
        place_prob = probabilities[:top_k].sum()
        place_odds[horse] = 1 / place_prob if place_prob > 0 else np.inf

    # Output the results
    print("\nExpected Rivals Beaten Percentage (ERB%) for each horse (Gaussian Model):")
    for horse in decimal_odds_input.keys():
        print(f"{horse}: ERB% = {ERB_percentage[horse]:.2f}%")

    print(f"\nPlace Probabilities and Place Odds for each horse (finishing in top {top_k} positions):")
    for horse in decimal_odds_input.keys():
        place_prob = place_probabilities[horse][:top_k].sum()
        print(f"{horse}: Place Probability = {place_prob:.4f}, Place Odds = {place_odds[horse]:.2f}")
