
import pandas as pd
import os
import numpy as np
import logging

# === Logging setup ===
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('build_features_kharif_models')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, "build_features_kharif_models.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



# === Constants ===
SEASON = "kharif"

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
    

def get_model_features():
    """Define feature columns for each model."""
    return {
        "surface_water_area_in_swb": [
            'uid', 'rainfall', 'runoff', 'dryspell_length', 'temp_extreme_days_proportion',
            'log_proportional_sw_area_over_cropping_kharif_prev', 'log_frac_sw_area_over_cropping_kharif_prev',
            'log_norm_surface_water_area_in_swb_kharif_prev', 'max_sw_area', 'isAlluviumAquifer'
        ],
        
        "area_under_hiz": [
            'uid', 'log_norm_surface_water_area_in_swb_kharif', 'rainfall', 'dryspell_length',
            'log_norm_area_wb_hiz_kharif_prev', 'log_norm_surface_water_area_in_swb_kharif_prev',
            'log_frac_area_hiz_over_sw_area_kharif_prev', 'max_area_hiz', 'isAlluviumAquifer'
        ],
        
        "area_under_liz": [
            'uid', 'rainfall', 'log_norm_area_wb_liz_kharif_prev',
            'log_frac_area_liz_over_sw_area_prev', 'log_norm_area_wb_hiz_kharif_prev',
            'log_norm_area_wb_hiz_kharif', 'max_area_liz', 'isAlluviumAquifer'
        ],
        
        "cropping_area_hiz": [
            'uid', 'rainfall', 'dryspell_length', 'log_norm_cropping_area_hiz_kharif_prev',
            'log_frac_cropping_area_hiz_over_sw_kharif_prev', 'log_frac_cropping_area_hiz_kharif_prev',
            'log_frac_area_hiz_over_sw_area_kharif', 'max_cropping_area_hiz', 'isAlluviumAquifer'
        ],
        
        "crop_health_hiz": [
            'uid', 'rainfall', 'runoff', 'rainfall_deviation_class', 'temp_extreme_intensity',
            'temp_extreme_days_proportion', 'log_norm_surface_water_area_in_swb_kharif', 'norm_area_wb_hiz_kharif',
            'crop_health_hiz_kharif_prev', 'crop_health_hiz_kharif_2017', 'isAlluviumAquifer'
        ],
        
        "cropping_area_liz": [
            'uid', 'rainfall', 'dryspell_length', 'log_norm_area_wb_liz_kharif',
            'log_norm_cropping_area_liz_kharif_prev', 'log_frac_cropping_area_liz_over_sw_kharif_prev',
            'log_proportional_cropping_area_liz_kharif_prev', 'log_frac_area_liz_over_sw_kharif',
            'gw_irrigation_liz_kharif_prev', 'log_mws_soge_value_prev', 'max_cropping_area_liz', 'isAlluviumAquifer'
        ],
        
        "cropping_area_other": [
            'uid', 'rainfall', 'dryspell_length', 'log_norm_area_wb_other_kharif',
            'log_norm_area_wb_other_kharif_prev', 'log_norm_cropping_area_other_kharif_prev',
            'max_cropping_area_other', 'gw_irrigation_other_prev', 'log_mws_soge_value_prev', 'isAlluviumAquifer'
        ],
        
        "crop_health_liz": [
            'uid', 'rainfall', 'runoff', 'monsoon_onset_dev_days', 'rainfall_deviation_class',
            'norm_surface_water_area_in_swb_kharif', 'frac_norm_cropping_area_liz_kharif_prev', 'crop_health_liz_kharif_prev',
            'gw_irrigation_liz_kharif_prev', 'log_mws_soge_value_prev', 'crop_health_liz_kharif_2017', 'isAlluviumAquifer'
        ],
        
        "crop_health_other": [
            'uid', 'log_norm_area_wb_other_kharif', 'monsoon_onset_dev_days', 'rainfall',
            'temp_extreme_days_proportion', 'dryspell_length', 'gw_irrigation_other_prev',
            'log_mws_soge_value_prev', 'isAlluviumAquifer'
        ],
        
        "abs_gw_liz": [
            'uid', 'norm_cropping_area_liz_kharif', 'crop_health_liz_kharif', 'rainfall',
            'temp_extreme_days_proportion', 'dryspell_length', 'rainfall_deviation_class', 'isAlluviumAquifer'
        ],
        
        "abs_gw_other": [
            'uid', 'norm_cropping_area_other_kharif', 'crop_health_other_kharif', 'rainfall',
            'temp_extreme_days_proportion', 'dryspell_length', 'rainfall_deviation_class', 'isAlluviumAquifer'
        ],
        
        "mws_soge": [
            'uid', 'groundwater_abstraction_other', 'groundwater_abstraction_other_prev',
            'log_mws_soge_value_prev', 'isAlluviumAquifer'
        ]
    }

def generate_model_features(df, split):
    """Generate features for all models by selecting appropriate columns."""
    model_features = get_model_features()
    
    # Remove first year data (no lagged features available)
    df_filtered = df.groupby('uid').apply(lambda x: x.iloc[1:]).reset_index(drop=True)
    logger.info(f"Filtered data for {split}: {len(df_filtered)} rows (removed first year for each UID)")
    
    for model_name, feature_cols in model_features.items():
        try:
            # Check which columns are available
            available_cols = [col for col in feature_cols if col in df_filtered.columns]
            missing_cols = [col for col in feature_cols if col not in df_filtered.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns for {model_name}: {missing_cols}")
            
            if available_cols:
                model_df = df_filtered[available_cols].copy()
                # Remove rows with any NaN values
                initial_rows = len(model_df)
                model_df = model_df.dropna()
                final_rows = len(model_df)
                
                if initial_rows != final_rows:
                    logger.info(f"Removed {initial_rows - final_rows} rows with NaN values for {model_name}")
                
                if len(model_df) > 0:
                    save_features(model_df, SEASON, split, model_name)
                else:
                    logger.warning(f"No valid data remaining for {model_name}")
            else:
                logger.error(f"No available columns for {model_name}")
                
        except Exception as e:
            logger.error(f"Error generating features for {model_name}: {e}")
            continue

def main():
    try:
        logger.info("Starting feature generation process...")
        
        # Load data
        logger.info("Loading data...")
        train_df, test_df = load_data(SEASON)
        logger.info(f"Loaded train data: {len(train_df)} rows, test data: {len(test_df)} rows")
        
        # Process training data
        logger.info("Processing training data...")
        train_df = compute_normalized_features(train_df)
        train_df = create_lagged_features(train_df)
        train_df = compute_derived_features(train_df)
        generate_model_features(train_df, "train")
        
        # Process test data
        logger.info("Processing test data...")
        test_df = compute_normalized_features(test_df)
        test_df = create_lagged_features(test_df)
        test_df = compute_derived_features(test_df)
        generate_model_features(test_df, "test")
        
        logger.info("Feature generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error in main process: {e}")
        raise

if __name__ == "__main__":
    main()