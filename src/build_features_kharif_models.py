
import pandas as pd
import os
import numpy as np
import logging


# === Constants ===
SEASON = "kharif"

# === Logging setup ===
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger(f'build_features_{SEASON}_models')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, f"build_features_{SEASON}_models.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# === Data IO ===
def load_data(season: str):
    train = pd.read_csv(f"./data/interim/{season}/train.csv")
    test = pd.read_csv(f"./data/interim/{season}/test.csv")
    return train, test

def save_features(df: pd.DataFrame, season: str, split: str, model_name: str):
    path = f"./data/features/{season}/{split}/"
    os.makedirs(path, exist_ok=True)
    df.to_csv(os.path.join(path, f"{model_name}.csv"), index=False)
    logger.info(f"Saved features for {model_name} - {split}: {len(df)} rows")

def save_targets(df: pd.DataFrame, season: str, split: str, model_name: str):
    """Save target values for a specific model."""
    path = f"./data/targets/{season}/{split}/"
    os.makedirs(path, exist_ok=True)
    df.to_csv(os.path.join(path, f"{model_name}.csv"), index=False)
    logger.info(f"Saved targets for {model_name} - {split}: {len(df)} rows")


# === Feature Engineering Functions ===
def create_lagged_features(df):
    """Create lagged features by shifting values within each UID group."""
    logger.info("Creating lagged features...")
    
    df = df.sort_values(['uid', 'year']).copy()

    # Define columns to lag
    lag_columns = [
        'norm_surface_water_area_in_swb_kharif',
        'total_cropping_area_kharif', 'norm_area_wb_hiz_kharif', 'norm_area_wb_liz_kharif',
        'norm_area_wb_other_kharif', 'norm_cropping_area_hiz_kharif', 
        'norm_cropping_area_liz_kharif', 'norm_cropping_area_other_kharif'
    ]
    
    # Create lagged features
    for col in lag_columns:
        if col in df.columns:
            df[f'{col}_prev'] = df.groupby('uid')[col].shift(1)
        else:
            logger.warning(f"Column {col} not found in dataframe")
    
    # Add specific lagged columns that are referenced in derived features
    additional_lag_columns = [
        'area_wb_hiz_kharif', 'area_wb_liz_kharif', 'surface_water_area_in_swb_kharif',
        'mws_soge_value', 'gw_irrigation_liz_kharif', 'gw_irrigation_other_kharif',
        'crop_health_hiz_kharif', 'crop_health_liz_kharif'
    ]
    
    for col in additional_lag_columns:
        if col in df.columns:
            df[f'{col}_prev'] = df.groupby('uid')[col].shift(1)
        else:
            logger.warning(f"Column {col} not found in dataframe")
    
    logger.info("Successfully computed lagged features...")
    return df

def compute_normalized_features(df):
    """Compute the basic normalized features at once."""
    logger.info("Computing normalized features...")
    try: 
        # Normalize features
        df['norm_surface_water_area_in_swb_kharif'] = df['surface_water_area_in_swb_kharif'] / np.clip(df['max_sw_area'], 1e-10, None)
        df['norm_area_wb_hiz_kharif'] = df['area_wb_hiz_kharif'] / np.clip(df['max_area_hiz'], 1e-10, None)
        df['norm_area_wb_liz_kharif'] = df['area_wb_liz_kharif'] / np.clip(df['max_area_liz'], 1e-10, None)
        df['norm_area_wb_other_kharif'] = df['area_wb_other_kharif'] / np.clip(df['max_area_other'], 1e-10, None)
        df['norm_cropping_area_hiz_kharif'] = df['cropping_area_hiz_kharif'] / np.clip(df['max_cropping_area_hiz'], 1e-10, None)
        df['norm_cropping_area_liz_kharif'] = df['cropping_area_liz_kharif'] / np.clip(df['max_cropping_area_liz'], 1e-10, None)
        df['norm_cropping_area_other_kharif'] = df['cropping_area_other_kharif'] / np.clip(df['max_cropping_area_other'], 1e-10, None)
        
        # Define the columns with normalized values
        # norm_columns = [
        #     'norm_surface_water_area_in_swb_kharif',
        #     'norm_area_wb_hiz_kharif',
        #     'norm_area_wb_liz_kharif',
        #     'norm_area_wb_other_kharif',
        #     'norm_cropping_area_hiz_kharif',
        #     'norm_cropping_area_liz_kharif',
        #     'norm_cropping_area_other_kharif'
        # ]

        # # Drop rows where **any** normalized value is greater than 1
        # df = df[~(df[norm_columns] > 1).any(axis=1)]
        logger.info("Successfully computed normalized features...")
        return df
    except Exception as e:
        logger.error(f"Error computing normalized features: {e}")
        raise

def compute_derived_features(df):
    """Compute all derived features (log_norm, norm, etc.) at once."""
    logger.info("Computing derived features...")
    
    try:
        # Log-normalized features
        df['log_norm_surface_water_area_in_swb_kharif'] = np.log1p(df['norm_surface_water_area_in_swb_kharif'])
        df['log_norm_area_wb_hiz_kharif'] = np.log1p(df['norm_area_wb_hiz_kharif'])
        df['log_norm_area_wb_liz_kharif'] = np.log1p(df['norm_area_wb_liz_kharif'])
        df['log_norm_area_wb_other_kharif'] = np.log1p(df['norm_area_wb_other_kharif'])
        df['log_norm_surface_water_area_in_swb_kharif_prev'] = np.log1p(df['norm_surface_water_area_in_swb_kharif_prev'])
        df['log_norm_area_wb_hiz_kharif_prev'] = np.log1p(df['norm_area_wb_hiz_kharif_prev'])
        df['log_norm_area_wb_liz_kharif_prev'] = np.log1p(df['norm_area_wb_liz_kharif_prev'])
        df['log_norm_area_wb_other_kharif_prev'] = np.log1p(df['norm_area_wb_other_kharif_prev'])

        # Cropping area log features
        df['log_norm_cropping_area_hiz_kharif'] = np.log1p(df['norm_cropping_area_hiz_kharif'])
        df['log_norm_cropping_area_liz_kharif'] = np.log1p(df['norm_cropping_area_liz_kharif'])
        df['log_norm_cropping_area_other_kharif'] = np.log1p(df['norm_cropping_area_other_kharif'])
        df['log_norm_cropping_area_hiz_kharif_prev'] = np.log1p(df['norm_cropping_area_hiz_kharif_prev'])
        df['log_norm_cropping_area_liz_kharif_prev'] = np.log1p(df['norm_cropping_area_liz_kharif_prev'])
        df['log_norm_cropping_area_other_kharif_prev'] = np.log1p(df['norm_cropping_area_other_kharif_prev'])
        
        # Fractional and proportional features
        df['log_frac_sw_area_over_cropping_kharif_prev'] = np.log1p(df['norm_surface_water_area_in_swb_kharif_prev'] / np.clip(df['max_cropping_area'], 1e-10, None))
        df['log_proportional_sw_area_over_cropping_kharif_prev'] = np.log1p(df['norm_surface_water_area_in_swb_kharif_prev'] / np.clip(df['total_cropping_area_kharif_prev'], 1e-10, None))
        
        # HIZ specific features
        df['log_frac_area_hiz_over_sw_area_kharif'] = np.log1p(df['area_wb_hiz_kharif'] / np.clip(df['surface_water_area_in_swb_kharif'], 1e-10, None))
        df['log_frac_area_hiz_over_sw_area_kharif_prev'] = np.log1p(df['area_wb_hiz_kharif_prev'] / np.clip(df['surface_water_area_in_swb_kharif_prev'], 1e-10, None))
        
        # LIZ specific features
        df['log_frac_area_liz_over_sw_area_kharif_prev'] = np.log1p(df['area_wb_liz_kharif_prev'] / np.clip(df['surface_water_area_in_swb_kharif_prev'], 1e-10, None))
        
        # Cropping area fractional features
        df['log_frac_cropping_area_hiz_over_sw_kharif_prev'] = np.log1p(df['norm_cropping_area_hiz_kharif_prev'] / np.clip(df['norm_surface_water_area_in_swb_kharif_prev'], 1e-10, None))
        df['log_frac_cropping_area_hiz_kharif_prev'] = np.log1p(df['norm_cropping_area_hiz_kharif_prev'] / np.clip(df['norm_area_wb_hiz_kharif_prev'], 1e-10, None))

        # LIZ cropping features
        df['log_frac_cropping_area_liz_over_sw_kharif_prev'] = np.log1p(df['norm_cropping_area_liz_kharif_prev'] / np.clip(df['norm_surface_water_area_in_swb_kharif_prev'], 1e-10, None))
        df['log_proportional_cropping_area_liz_kharif_prev'] = np.log1p(df['norm_cropping_area_liz_kharif_prev'] / np.clip(df['norm_area_wb_liz_kharif_prev'], 1e-10, None))
        df['log_frac_area_liz_over_sw_kharif'] = np.log1p(df['norm_area_wb_liz_kharif_prev'] / np.clip(df['norm_surface_water_area_in_swb_kharif_prev'], 1e-10, None))
        df['log_proportional_norm_cropping_area_hiz_kharif'] = np.log1p(df['norm_cropping_area_hiz_kharif'] / np.clip(df['norm_area_wb_hiz_kharif'], 1e-10, None))
        
        # Other area features
        df['log_proportional_cropping_area_other_kharif_prev'] = np.log1p(df['norm_cropping_area_other_kharif_prev'] / np.clip(df['norm_area_wb_other_kharif_prev'], 1e-10, None))
        
        # Groundwater and SOGE features
        df['log_mws_soge_value_prev'] = np.log1p(df['mws_soge_value_prev'])
        df['frac_norm_cropping_area_liz_kharif_prev'] = df['norm_cropping_area_liz_kharif_prev'] / np.clip(df['norm_area_wb_liz_kharif_prev'], 1e-10, None)
        
        logger.info("Successfully computed all derived features")
        return df
    except Exception as e:
        logger.error(f"Error computing derived features: {e}")
        raise

def compute_normalized_targets(df):
    """Compute normalized target variables and save normalization factors."""
    logger.info("Computing normalized targets...")
    
    try:
        # Normalize surface water area target
        df['norm_factor_surface_water_area_in_swb_kharif'] = np.clip(df['max_sw_area'], 1e-10, None)
        df['target_norm_surface_water_area_in_swb_kharif'] = df['surface_water_area_in_swb_kharif'] / df['norm_factor_surface_water_area_in_swb_kharif']
        
        # Normalize area targets
        df['norm_factor_area_wb_hiz_kharif'] = np.clip(df['max_area_hiz'], 1e-10, None)
        df['target_norm_area_wb_hiz_kharif'] = df['area_wb_hiz_kharif'] / df['norm_factor_area_wb_hiz_kharif']
        
        df['norm_factor_area_wb_liz_kharif'] = np.clip(df['max_area_liz'], 1e-10, None)
        df['target_norm_area_wb_liz_kharif'] = df['area_wb_liz_kharif'] / df['norm_factor_area_wb_liz_kharif']
        
        df['norm_factor_area_wb_other_kharif'] = np.clip(df['max_area_other'], 1e-10, None)
        df['target_norm_area_wb_other_kharif'] = df['area_wb_other_kharif'] / df['norm_factor_area_wb_other_kharif']
        
        # Normalize cropping area targets
        df['norm_factor_cropping_area_hiz_kharif'] = np.clip(df['max_cropping_area_hiz'], 1e-10, None)
        df['target_norm_cropping_area_hiz_kharif'] = df['cropping_area_hiz_kharif'] / df['norm_factor_cropping_area_hiz_kharif']
        
        df['norm_factor_cropping_area_liz_kharif'] = np.clip(df['max_cropping_area_liz'], 1e-10, None)
        df['target_norm_cropping_area_liz_kharif'] = df['cropping_area_liz_kharif'] / df['norm_factor_cropping_area_liz_kharif']
        
        df['norm_factor_cropping_area_other_kharif'] = np.clip(df['max_cropping_area_other'], 1e-10, None)
        df['target_norm_cropping_area_other_kharif'] = df['cropping_area_other_kharif'] / df['norm_factor_cropping_area_other_kharif']
        
        # Crop health targets (already normalized between 0-1, so no division needed)
        df['norm_factor_crop_health_hiz_kharif'] = 1.0
        df['target_crop_health_hiz_kharif'] = df['crop_health_hiz_kharif']
        
        df['norm_factor_crop_health_liz_kharif'] = 1.0
        df['target_crop_health_liz_kharif'] = df['crop_health_liz_kharif']
        
        df['norm_factor_crop_health_other_kharif'] = 1.0
        df['target_crop_health_other_kharif'] = df['crop_health_other_kharif']
        
        # Groundwater abstraction targets (normalized by area_in_ha)
        # df['norm_factor_gw_irrigation_liz_kharif'] = np.clip(df['area_in_ha'], 1e-10, None)
        df['norm_factor_gw_irrigation_liz_kharif'] = 1.0
        df['target_norm_gw_irrigation_liz_kharif'] = df['gw_irrigation_liz_kharif'] / df['norm_factor_gw_irrigation_liz_kharif']
        
        # df['norm_factor_gw_irrigation_other_kharif'] = np.clip(df['area_in_ha'], 1e-10, None)
        df['norm_factor_gw_irrigation_other_kharif'] = 1.0
        df['target_norm_gw_irrigation_other_kharif'] = df['gw_irrigation_other_kharif'] / df['norm_factor_gw_irrigation_other_kharif']
        
        # MWS SOGE target (no log transformation, keep original)
        df['norm_factor_mws_soge_value'] = 1.0
        df['target_mws_soge_value'] = df['mws_soge_value']

        logger.info("Successfully computed normalized targets")
        return df
        
    except Exception as e:
        logger.error(f"Error computing normalized targets: {e}")
        raise

def get_model_config():
    """Define feature columns and target configuration for each model."""
    return {
        "surface_water_area_in_swb": {
            'features': [
                'uid', 'rainfall_kharif', 'runoff_kharif', 'dryspell_length', 'temp_extreme_days_proportion_kharif',
                'log_proportional_sw_area_over_cropping_kharif_prev', 'log_frac_sw_area_over_cropping_kharif_prev',
                'log_norm_surface_water_area_in_swb_kharif_prev', 'isAlluviumAquifer'
            ],
            'target_column': 'target_norm_surface_water_area_in_swb_kharif',
            'norm_factor_column': 'norm_factor_surface_water_area_in_swb_kharif',
            'id_columns': ['uid', 'year']
        },
        
        "area_under_hiz": {
            'features': [
                'uid', 'log_norm_surface_water_area_in_swb_kharif', 'rainfall_kharif', 'dryspell_length',
                'log_norm_area_wb_hiz_kharif_prev', 'log_norm_surface_water_area_in_swb_kharif_prev',
                'log_frac_area_hiz_over_sw_area_kharif_prev', 'isAlluviumAquifer'
            ],
            'target_column': 'target_norm_area_wb_hiz_kharif',
            'norm_factor_column': 'norm_factor_area_wb_hiz_kharif',
            'id_columns': ['uid', 'year']
        },
        
        "area_under_liz": {
            'features': [
                'uid', 'rainfall_kharif', 'log_norm_area_wb_liz_kharif_prev',
                'log_frac_area_liz_over_sw_area_kharif_prev', 'log_norm_area_wb_hiz_kharif_prev',
                'log_norm_area_wb_hiz_kharif', 'isAlluviumAquifer'
            ],
            'target_column': 'target_norm_area_wb_liz_kharif',
            'norm_factor_column': 'norm_factor_area_wb_liz_kharif',
            'id_columns': ['uid', 'year']
        },
        
        "cropping_area_hiz": {
            'features': [
                'uid', 'rainfall_kharif', 'dryspell_length', 'log_norm_cropping_area_hiz_kharif_prev',
                'log_frac_cropping_area_hiz_over_sw_kharif_prev', 'log_frac_cropping_area_hiz_kharif_prev',
                'log_frac_area_hiz_over_sw_area_kharif', 'isAlluviumAquifer'
            ],
            'target_column': 'target_norm_cropping_area_hiz_kharif',
            'norm_factor_column': 'norm_factor_cropping_area_hiz_kharif',
            'id_columns': ['uid', 'year']
        },
        
        "crop_health_hiz": {
            'features': [
                'uid', 'rainfall_kharif', 'runoff_kharif', 'rainfall_deviation_class', 'temp_extreme_intensity_kharif',
                'temp_extreme_days_proportion_kharif', 'log_norm_surface_water_area_in_swb_kharif', 'norm_area_wb_hiz_kharif',
                'crop_health_hiz_kharif_prev', 'crop_health_hiz_kharif_2017', 'isAlluviumAquifer'
            ],
            'target_column': 'target_crop_health_hiz_kharif',
            'norm_factor_column': 'norm_factor_crop_health_hiz_kharif',
            'id_columns': ['uid', 'year']
        },
        
        "cropping_area_liz": {
            'features': [
                'uid', 'rainfall_kharif', 'dryspell_length', 'log_norm_area_wb_liz_kharif',
                'log_norm_cropping_area_liz_kharif_prev', 'log_frac_cropping_area_liz_over_sw_kharif_prev',
                'log_proportional_cropping_area_liz_kharif_prev', 'log_frac_area_liz_over_sw_kharif',
                'gw_irrigation_liz_kharif_prev', 'log_mws_soge_value_prev', 'isAlluviumAquifer'
            ],
            'target_column': 'target_norm_cropping_area_liz_kharif',
            'norm_factor_column': 'norm_factor_cropping_area_liz_kharif',
            'id_columns': ['uid', 'year']
        },
        
        "cropping_area_other": {
            'features': [
                'uid', 'rainfall_kharif', 'dryspell_length', 'log_norm_area_wb_other_kharif',
                'log_norm_area_wb_other_kharif_prev', 'log_norm_cropping_area_other_kharif_prev',
                'gw_irrigation_other_kharif_prev', 'log_mws_soge_value_prev', 'isAlluviumAquifer'
            ],
            'target_column': 'target_norm_cropping_area_other_kharif',
            'norm_factor_column': 'norm_factor_cropping_area_other_kharif',
            'id_columns': ['uid', 'year']
        },
        
        "crop_health_liz": {
            'features': [
                'uid', 'rainfall_kharif', 'runoff_kharif', 'monsoon_onset_dev_days', 'rainfall_deviation_class',
                'norm_surface_water_area_in_swb_kharif', 'frac_norm_cropping_area_liz_kharif_prev', 'crop_health_liz_kharif_prev',
                'gw_irrigation_liz_kharif_prev', 'log_mws_soge_value_prev', 'crop_health_liz_kharif_2017', 'isAlluviumAquifer'
            ],
            'target_column': 'target_crop_health_liz_kharif',
            'norm_factor_column': 'norm_factor_crop_health_liz_kharif',
            'id_columns': ['uid', 'year']
        },
        
        "crop_health_other": {
            'features': [
                'uid', 'log_norm_area_wb_other_kharif', 'monsoon_onset_dev_days', 'rainfall_kharif',
                'temp_extreme_days_proportion_kharif', 'dryspell_length', 'gw_irrigation_other_kharif_prev',
                'log_mws_soge_value_prev', 'isAlluviumAquifer'
            ],
            'target_column': 'target_crop_health_other_kharif',
            'norm_factor_column': 'norm_factor_crop_health_other_kharif',
            'id_columns': ['uid', 'year']
        },
        
        "abs_gw_liz": {
            'features': [
                'uid', 'norm_cropping_area_liz_kharif', 'crop_health_liz_kharif', 'rainfall_kharif',
                'temp_extreme_days_proportion_kharif', 'dryspell_length', 'rainfall_deviation_class', 'isAlluviumAquifer'
            ],
            'target_column': 'target_norm_gw_irrigation_liz_kharif',
            'norm_factor_column': 'norm_factor_gw_irrigation_liz_kharif',
            'id_columns': ['uid', 'year']
        },
        
        "abs_gw_other": {
            'features': [
                'uid', 'norm_cropping_area_other_kharif', 'crop_health_other_kharif', 'rainfall_kharif',
                'temp_extreme_days_proportion_kharif', 'dryspell_length', 'rainfall_deviation_class', 'isAlluviumAquifer'
            ],
            'target_column': 'target_norm_gw_irrigation_other_kharif',
            'norm_factor_column': 'norm_factor_gw_irrigation_other_kharif',
            'id_columns': ['uid', 'year']
        },
        
        "mws_soge": {
            'features': [
                'uid', 'gw_irrigation_liz_kharif', 'gw_irrigation_other_kharif', 'gw_irrigation_liz_kharif_prev',
                'gw_irrigation_other_kharif_prev', 'log_mws_soge_value_prev', 'isAlluviumAquifer'
            ],
            'target_column': 'target_mws_soge_value',
            'norm_factor_column': 'norm_factor_mws_soge_value',
            'id_columns': ['uid', 'year']
        }
    }


def generate_model_data(df, split):
    """Generate both features and targets for all models consistently."""
    model_config = get_model_config()

    for model_name, config in model_config.items():
        try:
            feature_cols = config['features']
            target_col = config['target_column']
            norm_factor_col = config['norm_factor_column']
            id_cols = config['id_columns']

            # Combine all required columns for this model
            required_cols = list(set(feature_cols + id_cols + [target_col, norm_factor_col]))
            model_df = df[required_cols].copy()
            
            # Drop rows where **any** target-normalized value > 1
            excluded_prefixes = [
                "target_mws_soge_value",
                "target_norm_gw_irrigation_other_",
                "target_norm_gw_irrigation_liz_"
            ]
            target_norm_cols = [
                col for col in model_df.columns
                if col.startswith("target_") and not any(col.startswith(prefix) for prefix in excluded_prefixes)
            ]
            if target_norm_cols:
                model_df = model_df[~(model_df[target_norm_cols] > 1).any(axis=1)]

            norm_columns = [col for col in feature_cols if "norm" in col and "factor" not in col and "target" not in col]
            # # Drop rows where **any** normalized value is greater than 1
            model_df = model_df[~(model_df[norm_columns] > 1).any(axis=1)]

            # Drop rows with any NaNs across all required columns
            initial_rows = len(model_df)
            model_df = model_df.dropna()
            final_rows = len(model_df)

            if final_rows == 0:
                logger.warning(f"No valid rows remaining after filtering for {model_name}")
                continue

            logger.info(f"{model_name}: Removed {initial_rows - final_rows} rows with missing data")

            # Save features and targets from the cleaned dataframe
            feature_df = model_df[feature_cols]
            target_df = model_df[id_cols + [target_col, norm_factor_col]]

            save_features(feature_df, SEASON, split, model_name)
            save_targets(target_df, SEASON, split, model_name)

        except Exception as e:
            logger.error(f"Error processing model {model_name}: {e}")
            continue


# === Main execution function ===
def main():
    """Main function to orchestrate the feature engineering process."""
    logger.info(f"Starting feature engineering for {SEASON} season...")
    
    try:
        # Load data
        logger.info("Loading data...")
        train_df, test_df = load_data(SEASON)
        logger.info(f"Loaded train data: {len(train_df)} rows, test data: {len(test_df)} rows")
        
        # Process training data
        logger.info("Processing training data...")
        train_df = compute_normalized_features(train_df)
        train_df = create_lagged_features(train_df)
        train_df = compute_derived_features(train_df)
        train_df = compute_normalized_targets(train_df)
        
        # Generate model data for training
        logger.info("Generating model data for training...")
        generate_model_data(train_df, "train")
        
        # Process test data
        logger.info("Processing test data...")
        test_df = compute_normalized_features(test_df)
        test_df = create_lagged_features(test_df)
        test_df = compute_derived_features(test_df)
        test_df = compute_normalized_targets(test_df)
        
        # Generate model data for testing
        logger.info("Generating model data for testing...")
        generate_model_data(test_df, "test")
        
        logger.info("Feature engineering completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()