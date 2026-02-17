# generate_dataset.py
import gc
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Edit this to adjust the size according to your machine's memory
TRAIN_NUM_FILES = 1
TEST_NUM_FILES = 5

# DO NOT EDIT THE CODE BELOW
NUM_USERS = 1_000_000
NUM_RESTAURANTS = 100_000
NUM_REQUESTS = 10_000
NUM_CANDIDATES = 1000
USER_FEATURES = 30
RESTAURANT_FEATURES = 10
MIN_LAT = 13.581327544528989
MAX_LAT = 14.171447685944495
MIN_LNG = 100.30471258549801
MAX_LNG = 100.85824174657563
TRAIN_ROWS = 50_000_000
TEST_ROWS = 5_000_000

OUTPUT_DIR = Path("data")
os.makedirs(OUTPUT_DIR / "train", exist_ok=True)
os.makedirs(OUTPUT_DIR / "test", exist_ok=True)


def generate_gaussian_features(n, dim, clusters=10, std=0.1):
    """
    Generate multivariate Gaussian features centered around cluster centroids.
    """
    cluster_centers = np.random.randn(clusters, dim) * 2
    assignments = np.random.choice(clusters, size=n)
    features = np.array(
        [
            np.random.normal(loc=cluster_centers[c], scale=std, size=dim)
            for c in assignments
        ]
    )
    return features.astype(np.float32)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_click_data(
    num_rows, user_features, restaurant_features
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    data = []
    for _ in range(num_rows):
        user_id = np.random.randint(NUM_USERS)
        rest_id = np.random.randint(NUM_RESTAURANTS)

        user_vec = user_features[user_id]
        rest_vec = restaurant_features[rest_id]

        # Click score: dot product + noise
        score = np.dot(
            user_vec, np.pad(rest_vec, (0, USER_FEATURES - RESTAURANT_FEATURES))
        ) + np.random.normal(0, 0.1)
        prob = sigmoid(score)
        click = np.random.binomial(1, prob)

        row = [user_id, rest_id, click]
        data.append(row)

    columns = ["user_id", "restaurant_id", "click"]
    return pd.DataFrame(data, columns=columns)


def generate_user():
    print("Generating user features...")
    user_features = generate_gaussian_features(NUM_USERS, USER_FEATURES)
    user_df = pd.DataFrame(
        np.hstack((np.arange(user_features.shape[0]).reshape(-1, 1), user_features)),
        columns=["user_id"] + [f"f{ind:02}" for ind in range(user_features.shape[1])],
    )
    user_df["user_id"] = user_df["user_id"].astype(int)
    user_df.to_parquet(OUTPUT_DIR / "user_features.parquet")
    return user_df


def generate_restaurant():
    print("Generating restaurant features...")
    restaurant_features = generate_gaussian_features(
        NUM_RESTAURANTS, RESTAURANT_FEATURES
    )
    restaurant_df = pd.DataFrame(
        np.hstack(
            (
                np.arange(restaurant_features.shape[0]).reshape(-1, 1),
                np.random.uniform(
                    low=MIN_LAT, high=MAX_LAT, size=(restaurant_features.shape[0], 1)
                ),
                np.random.uniform(
                    low=MIN_LNG, high=MAX_LNG, size=(restaurant_features.shape[0], 1)
                ),
                restaurant_features,
            )
        ),
        columns=["restaurant_id", "latitude", "longitude"]
        + [f"f{ind:02}" for ind in range(restaurant_features.shape[1])],
    )
    restaurant_df["restaurant_id"] = restaurant_df["restaurant_id"].astype(int)
    restaurant_df.to_parquet(OUTPUT_DIR / "restaurant_features.parquet")
    return restaurant_df


def generate_activity_data(user_df, restaurant_df):
    user_features = user_df.drop(columns=["user_id"]).values
    restaurant_features = restaurant_df.drop(
        columns=["restaurant_id", "latitude", "longitude"]
    ).values
    train_rows_per_chunk = TRAIN_ROWS // TRAIN_NUM_FILES
    for i in tqdm(range(TRAIN_NUM_FILES), desc="train set"):
        df = generate_click_data(
            train_rows_per_chunk, user_features, restaurant_features
        )
        df.to_parquet(OUTPUT_DIR / "train" / f"train_{i:02}.parquet")

    # Generate test set
    test_rows_per_chunk = TEST_ROWS // TEST_NUM_FILES
    print("Loading test set...")
    for i in tqdm(range(TEST_NUM_FILES), desc="test set"):
        test_df = generate_click_data(
            test_rows_per_chunk, user_features, restaurant_features
        )
        test_df.to_parquet(OUTPUT_DIR / "test" / f"test_{i:02}.parquet")


def generate_requests(user_df: pd.DataFrame, restaurant_df: pd.DataFrame, size=10000):
    print("Generating request data...")
    sampled_user_ids = user_df["user_id"].sample(n=size).values.reshape(-1, 1)

    sampled_restaurants_list = [
        restaurant_df["restaurant_id"].sample(n=NUM_CANDIDATES).to_list()
        for _ in range(size)
    ]
    sampled_restaurants = np.empty(size, object)
    sampled_restaurants[:] = sampled_restaurants_list
    sampled_restaurants = sampled_restaurants.reshape(-1, 1)
    df = pd.DataFrame(
        np.hstack(
            (
                sampled_user_ids,
                np.random.uniform(low=MIN_LAT, high=MAX_LAT, size=(size, 1)),
                np.random.uniform(low=MIN_LNG, high=MAX_LNG, size=(size, 1)),
                sampled_restaurants,
                np.random.randint(low=1, high=11, size=(size, 1)) * 100,
                np.random.randint(low=5, high=11, size=(size, 1)) * 1000,
                np.random.randint(low=0, high=2, size=(size, 1)).astype(bool),
            )
        ),
        columns=[
            "user_id",
            "latitude",
            "longitude",
            "candidate_restaurant_ids",
            "size",
            "max_dist",
            "sort_dist",
        ],
    )
    df.to_parquet("data/requests.parquet")


if __name__ == "__main__":
    np.random.seed(7)
    user_df = generate_user()
    restaurant_df = generate_restaurant()
    generate_activity_data(user_df, restaurant_df)
    generate_requests(user_df, restaurant_df, size=NUM_REQUESTS)
    print("Done")
