# generate_initial_data.py
import random
import numpy as np
import json
import math

NUM_CONTENT = 25  # Number of unique content items
OUTPUT_FILE = 'initial_content_data.json'
MAX_SCORE_CAP = 1000 # Optional cap for calculated scores
# Log-normal parameters for actual_viewers (tune sigma for spread)
# Mean of the underlying normal distribution (log scale)
VIEWERS_LOG_MEAN = 7.0
# Standard deviation of the underlying normal distribution (log scale)
VIEWERS_LOG_SIGMA = 1.5

# Beta distribution parameters (alpha, beta) for rates (a<b means skewed towards 0)
LIKE_RATE_PARAMS = (1.5, 20)
COMMENT_RATE_PARAMS = (0.8, 50)
SAVE_RATE_PARAMS = (1.0, 40)
SHARE_RATE_PARAMS = (0.7, 60)

def calculate_engagement_score(metrics: dict) -> float:
    """
    Calculates engagement score based on the provided formula.

    Engagement Score = ((C * 0.1 + D * 0.5 + E * 1.5 + F * 0.5 + G * 1.5) / H) * 250

    Args:
        metrics (dict): Dictionary containing 'views', 'likes', 'comments',
                        'saves', 'shares', 'actual_viewers'.

    Returns:
        float: Calculated engagement score.
    """
    C = metrics.get('views', 0)
    D = metrics.get('likes', 0)
    E = metrics.get('comments', 0)
    F = metrics.get('saves', 0)
    G = metrics.get('shares', 0)
    H = metrics.get('actual_viewers', 1) # Use 1 to avoid division by zero

    if H <= 0:
        # Handle edge case where actual viewers might be zero or negative
        # print("Warning: Actual viewers (H) is zero or less. Setting score to 0.")
        return 0.0

    numerator = (C * 0.1 + D * 0.5 + E * 1.5 + F * 0.5 + G * 1.5)
    score = (numerator / H) * 250

    # Ensure score is not negative (shouldn't happen with positive inputs)
    score = max(0.0, score)

    # Optional: Cap the score
    score = min(score, MAX_SCORE_CAP)

    return score

def generate_content_item(content_id: int) -> dict:
    """
    Generates more realistic initial metrics using distributions and calculates engagement score.
    """
    # 1. Generate actual viewers using log-normal distribution
    # Add 1 to ensure at least 1 viewer, clip potential outliers if needed
    actual_viewers = max(1, int(np.random.lognormal(mean=VIEWERS_LOG_MEAN, sigma=VIEWERS_LOG_SIGMA) + 1))

    # 2. Generate views (often higher than viewers due to rewatches)
    # Using Gamma distribution: Shape > 1 means peak > 0, scale influences spread.
    # Adding 1 ensures views >= actual_viewers.
    rewatch_factor = np.random.gamma(shape=2, scale=0.8) # Average rewatch factor around 1.6
    views = max(actual_viewers, int(actual_viewers * (1 + rewatch_factor)))

    # 3. Generate interaction rates using Beta distribution
    # Beta distribution naturally outputs values between 0 and 1
    like_rate = np.random.beta(LIKE_RATE_PARAMS[0], LIKE_RATE_PARAMS[1])
    comment_rate = np.random.beta(COMMENT_RATE_PARAMS[0], COMMENT_RATE_PARAMS[1])
    save_rate = np.random.beta(SAVE_RATE_PARAMS[0], SAVE_RATE_PARAMS[1])
    share_rate = np.random.beta(SHARE_RATE_PARAMS[0], SHARE_RATE_PARAMS[1])

    # 4. Calculate absolute metrics from rates
    likes = max(0, int(actual_viewers * like_rate))
    comments = max(0, int(actual_viewers * comment_rate))
    saves = max(0, int(actual_viewers * save_rate))
    shares = max(0, int(actual_viewers * share_rate))

    metrics = {
        'views': views,
        'likes': likes,
        'comments': comments,
        'saves': saves,
        'shares': shares,
        'actual_viewers': actual_viewers
    }

    initial_score = calculate_engagement_score(metrics)

    # Content size
    size_mb = random.randint(5, 50) # MB

    content_data = {
        'content_id': content_id,
        'size_mb': size_mb,
        'metrics': metrics,
        # Store current score separately for easy access/update later
        'current_score': initial_score
    }
    return content_data

# --- Main Generation ---
if __name__ == "__main__":
    all_content_data = []
    print(f"Generating initial data for {NUM_CONTENT} content items...")
    for i in range(NUM_CONTENT):
        item_data = generate_content_item(i)
        all_content_data.append(item_data)
        print(f"  Generated ID {i}: Score={item_data['current_score']:.2f}, Viewers={item_data['metrics']['actual_viewers']}")

    print(f"\nSaving data to {OUTPUT_FILE}...")
    try:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(all_content_data, f, indent=4)
        print("Data saved successfully.")
    except IOError as e:
        print(f"Error saving data to {OUTPUT_FILE}: {e}")