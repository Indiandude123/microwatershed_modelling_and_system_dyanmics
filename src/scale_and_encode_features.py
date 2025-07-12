# import pandas as pd
# import os
# import numpy as np
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# import joblib
# import logging

# # === Logging setup ===
# log_dir = "logs"
# os.makedirs(log_dir, exist_ok=True)

# logger = logging.getLogger('scale_and_encode_features')
# logger.setLevel(logging.DEBUG)

# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.DEBUG)

# log_file_path = os.path.join(log_dir, "scale_and_encode_features.log")
# file_handler = logging.FileHandler(log_file_path)
# file_handler.setLevel(logging.DEBUG)

# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
# file_handler.setFormatter(formatter)

# logger.addHandler(console_handler)
# logger.addHandler(file_handler)

# # === Constants ===
# SEASON = "kharif"
# RAW_DIR = f"./data/features/{SEASON}"
# PROCESSED_DIR = f"./data/processed/{SEASON}"
# SCALER_DIR = f"./artifacts/scalers/{SEASON}"
# ENCODER_DIR = f"./artifacts/encoders/{SEASON}"
# os.makedirs(SCALER_DIR, exist_ok=True)
# os.makedirs(ENCODER_DIR, exist_ok=True)

# # === Helpers ===

# def get_all_model_files(split):
#     dir_path = os.path.join(RAW_DIR, split)
#     return [f for f in os.listdir(dir_path) if f.endswith(".csv")]

# def scale_and_encode(train_df, test_df, model_name):
#     logger.info(f"Processing model: {model_name}")

#     all_df = pd.concat([train_df, test_df], axis=0)
    
#     # Identify column types
#     cat_cols = all_df.select_dtypes(include=["object", "category"]).columns.tolist()
#     num_cols = all_df.select_dtypes(include=[np.number]).columns.tolist()

#     # Don't scale target or UID columns
#     uid_cols = [col for col in ['uid', 'year'] if col in all_df.columns]
#     target_cols = [col for col in all_df.columns if "target" in col or "label" in col]  # extend if needed
#     exclude_cols = set(uid_cols + target_cols)

#     num_cols = [col for col in num_cols if col not in exclude_cols]

#     # === Label Encode categorical features ===
#     for col in cat_cols:
#         le = LabelEncoder()
#         le.fit(all_df[col].astype(str))
#         train_df[col] = le.transform(train_df[col].astype(str))
#         test_df[col] = le.transform(test_df[col].astype(str))

#         joblib.dump(le, os.path.join(ENCODER_DIR, f"{model_name}_{col}_le.pkl"))
#         logger.debug(f"Label encoder saved for {col} in {model_name}")

#     # === Standardize numerical features ===
#     scaler = StandardScaler()
#     scaler.fit(train_df[num_cols])
#     train_df[num_cols] = scaler.transform(train_df[num_cols])
#     test_df[num_cols] = scaler.transform(test_df[num_cols])

#     joblib.dump(scaler, os.path.join(SCALER_DIR, f"{model_name}_scaler.pkl"))
#     logger.debug(f"Scaler saved for {model_name}")

#     return train_df, test_df


# def main():
#     logger.info("Starting scaling and encoding process...")
#     for split in ["train", "test"]:
#         files = get_all_model_files(split)
#         for file in files:
#             model_name = file.replace(".csv", "")
#             path_train = os.path.join(RAW_DIR, "train", file)
#             path_test = os.path.join(RAW_DIR, "test", file)

#             if not os.path.exists(path_train) or not os.path.exists(path_test):
#                 logger.warning(f"Skipping {model_name} due to missing split files.")
#                 continue

#             df_train = pd.read_csv(path_train)
#             df_test = pd.read_csv(path_test)

#             df_train_scaled, df_test_scaled = scale_and_encode(df_train, df_test, model_name)

#             out_train_dir = os.path.join(PROCESSED_DIR, "train")
#             out_test_dir = os.path.join(PROCESSED_DIR, "test")
#             os.makedirs(out_train_dir, exist_ok=True)
#             os.makedirs(out_test_dir, exist_ok=True)

#             df_train_scaled.to_csv(os.path.join(out_train_dir, f"{model_name}.csv"), index=False)
#             df_test_scaled.to_csv(os.path.join(out_test_dir, f"{model_name}.csv"), index=False)

#             logger.info(f"Saved scaled features for {model_name}")


# if __name__ == "__main__":
#     main()

import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import logging

# === Logging setup ===
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('scale_and_encode_features')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, "scale_and_encode_features.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# === Constants ===
SEASON = "kharif"
RAW_DIR = f"./data/features/{SEASON}"
PROCESSED_DIR = f"./data/processed/{SEASON}"
SCALER_DIR = f"./artifacts/scalers/{SEASON}"
ENCODER_DIR = f"./artifacts/encoders/{SEASON}"
os.makedirs(SCALER_DIR, exist_ok=True)
os.makedirs(ENCODER_DIR, exist_ok=True)

# === Ordered categories for label encoding ===
rainfall_dev_order = ['Large Excess', 'Excess', 'Normal', 'Deficient', 'Large Deficient', 'No Rain']
spi_class_order = ["Extremely Dry", "Severely Dry", "Moderately Dry", "Mildly Dry",
                   "Mildly Wet", "Moderately Wet", "Severely Wet", "Extremely Wet"]

def encode_ordered_column(df, column, order, model_name):
    if column in df.columns:
        df[column] = pd.Categorical(df[column], categories=order, ordered=True).codes
        joblib.dump(order, os.path.join(ENCODER_DIR, f"{model_name}_{column}_order.pkl"))
        logger.debug(f"Label-encoded '{column}' using custom order for {model_name}")
    return df

# === Helpers ===
def get_all_model_files(split):
    dir_path = os.path.join(RAW_DIR, split)
    return [f for f in os.listdir(dir_path) if f.endswith(".csv")]

def scale_and_encode(train_df, test_df, model_name):
    logger.info(f"Processing model: {model_name}")

    all_df = pd.concat([train_df, test_df], axis=0)

    uid_cols = [col for col in ['uid', 'year', 'isAlluviumAquifer'] if col in all_df.columns]
    target_cols = [col for col in all_df.columns if "target" in col or "label" in col]
    exclude_cols = set(uid_cols + target_cols)

    # === Only encode these specific categorical features ===
    for col, order in {
        "rainfall_deviation_class": rainfall_dev_order,
        "spi_class": spi_class_order
    }.items():
        if col in all_df.columns:
            train_df = encode_ordered_column(train_df, col, order, model_name)
            test_df = encode_ordered_column(test_df, col, order, model_name)

    # === Standardize numerical features ===
    num_cols = all_df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [col for col in num_cols if col not in exclude_cols]

    scaler = StandardScaler()
    scaler.fit(train_df[num_cols])
    train_df[num_cols] = scaler.transform(train_df[num_cols])
    test_df[num_cols] = scaler.transform(test_df[num_cols])

    joblib.dump(scaler, os.path.join(SCALER_DIR, f"{model_name}_scaler.pkl"))
    logger.debug(f"Scaler saved for {model_name}")

    return train_df, test_df


def main():
    logger.info("Starting scaling and encoding process...")
    for split in ["train", "test"]:
        files = get_all_model_files(split)
        for file in files:
            model_name = file.replace(".csv", "")
            path_train = os.path.join(RAW_DIR, "train", file)
            path_test = os.path.join(RAW_DIR, "test", file)

            if not os.path.exists(path_train) or not os.path.exists(path_test):
                logger.warning(f"Skipping {model_name} due to missing split files.")
                continue

            df_train = pd.read_csv(path_train)
            df_test = pd.read_csv(path_test)

            df_train_scaled, df_test_scaled = scale_and_encode(df_train, df_test, model_name)

            out_train_dir = os.path.join(PROCESSED_DIR, "train")
            out_test_dir = os.path.join(PROCESSED_DIR, "test")
            os.makedirs(out_train_dir, exist_ok=True)
            os.makedirs(out_test_dir, exist_ok=True)

            df_train_scaled.to_csv(os.path.join(out_train_dir, f"{model_name}.csv"), index=False)
            df_test_scaled.to_csv(os.path.join(out_test_dir, f"{model_name}.csv"), index=False)

            logger.info(f"Saved scaled features for {model_name}")


if __name__ == "__main__":
    main()
