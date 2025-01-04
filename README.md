
# Horse Racing Prediction Model

## Overview
This repository contains a project focused on predicting the outcomes of horse races based on historical data from the Proform Racing database. The dataset used for training is focused on **2024 UK All Weather Horse Racing** and has been selected for its relevance to developing a model that can predict race outcomes, starting with a relatively simple handicap model for All Weather races.

## Dataset
The dataset is in CSV format and contains various features related to the races and horses, such as:
- Race-specific information (e.g., track, distance, race type)
- Horse-specific information (e.g., speed, days since last run, jockey rating)
- Additional variables like pace and draw ratings

You can access the glossary that explains the features in the dataset [here](https://www.proformracing.com/software-glossaries.html).

**Note:** The dataset has been pre-filtered to focus on All Weather races and selected features that are considered critical for model development.

## Problem Statement
Horse racing presents a range of challenges when it comes to modeling:
1. **Data consistency**: We need to ensure that the data reflects "as was" information rather than "as is" information, i.e., the race data should reflect the state of information at the time of the race and not updated post-race.
2. **Variable race sizes**: Different races may have different numbers of runners (up to a maximum of 20 in All Weather races). This may require padding or adjustments in the model.
3. **Complex feature engineering**: Features such as pace and draw ratings require additional calculations (e.g., calculating pace pressure based on pre-race pace figures).
4. **Race types**: Handicap races are expected to be easier to model initially as they involve weights that aim to equalize the chance of winning among competitors.

## Model Approach
We are exploring two potential modeling approaches for this project:

### 1. **Gradient Boosted Decision Trees (GBDTs) with Probability Calibration**
- **How it works**: Train a GBDT (e.g., LightGBM, XGBoost, or CatBoost) on a binary classification problem (Winner vs. Not Winner). Afterward, apply probability calibration (e.g., Platt scaling, isotonic regression) to ensure the output reflects true probabilities.
- **Pros**:
    - Excellent predictive performance on tabular data.
    - Handles large datasets efficiently.
    - Well-supported libraries (e.g., LightGBM, XGBoost).
- **Cons**:
    - Requires an extra step for probability calibration and normalization.
    - Comparisons between horses are implicit.

### 2. **Multinomial (Softmax) Approach**
- **How it works**: Build a model that outputs a probability distribution across all horses in a race, using a softmax layer to ensure the sum of probabilities equals 1.
- **Pros**:
    - The probabilities naturally sum to 1 for each race.
    - The model sees the entire set of competitors and learns to differentiate among all runners.
- **Cons**:
    - More complex implementation, especially handling races with fewer than 20 runners (padding or masking required).
    - Requires careful data preprocessing.

## Future Work
- Start with a simple All Weather handicap model, then progressively add complexity (e.g., non-handicap races, turf races).
- Further explore the impact of different features (e.g., pace pressure, draw) and their interaction.
- Investigate any inconsistencies in the Proform data (e.g., updating ratings post-race) and their potential impact on training.

## Installation
To set up this project locally, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/horse-racing-prediction.git
   cd horse-racing-prediction
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To train the model, use the following command:
```bash
python train_model.py
```
This will load the dataset, preprocess the data, and train a model based on the chosen approach.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
