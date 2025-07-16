import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('split_train_test.py')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "split_train_test.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_flattened_data(season: str) -> pd.DataFrame:
    """Load the flattened dataset for a given season."""
    try:
        path = f"./data/interim/flattened_dataset_{season}.csv"
        df = pd.read_csv(path)
        logger.info(f"Loaded {season} flattened data: shape = {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load {season} dataset: {e}")
        raise

def save_split_data(train_df: pd.DataFrame, test_df: pd.DataFrame, season: str):
    """Save the split train/test DataFrames to disk."""
    try:
        output_dir = f"./data/interim/{season}"
        os.makedirs(output_dir, exist_ok=True)

        train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

        logger.info(f"Saved train/test for {season} at {output_dir}")
    except Exception as e:
        logger.error(f"Saving failed for season {season}: {e}")
        raise

def split_by_year(df: pd.DataFrame):
    """Split the dataset into train and test based on the 'year' column."""
    train_years = [2018, 2019, 2020, 2021], 
    test_years = [2022]

    train_df = df[df["year"].isin(train_years)].copy()
    test_df = df[df["year"].isin(test_years)].copy()

    logger.info(f"Split: train={train_df.shape}, test={test_df.shape}")
    return train_df, test_df

def split_randomly(df: pd.DataFrame, test_size: int, random_state=42):
    """Split the dataset into train and test with every datapoint having uniform probability"""
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
    logger.info(f"📊 Random split: train = {train_df.shape}, test = {test_df.shape}")
    return train_df, test_df

def main():
    seasons = ["kharif", "rabi", "zaid"]
    split_type = "randomly"
    test_size = 0.25
    for season in seasons:
        try:
            df = load_flattened_data(season)
            if split_type == "year":
                train_df, test_df = split_by_year(df)
            else:
                train_df, test_df = split_randomly(df, test_size)
                
            save_split_data(train_df, test_df, season)
        except Exception:
            logger.exception(f"Skipping season {season} due to error.")

if __name__ == "__main__":
    main()
