
import os
import pandas as pd
import numpy as np
import joblib
import optuna
import logging
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

# === Paths ===
SEASON = "kharif"
TRAIN_FEATURES_DIR = f"./data/processed/{SEASON}/train"
TEST_FEATURES_DIR = f"./data/processed/{SEASON}/test"
TRAIN_TARGETS_DIR = f"./data/targets/{SEASON}/train"
TEST_TARGETS_DIR = f"./data/targets/{SEASON}/test"
MODEL_DIR = f"./artifacts/models/{SEASON}"
TUNING_DIR = f"./artifacts/hyperparams/{SEASON}"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TUNING_DIR, exist_ok=True)

# === Logging setup ===
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger(f'train_models_{SEASON}')
logger.setLevel(logging.DEBUG)

# Clear existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
log_file_path = os.path.join(log_dir, f"train_models_{SEASON}.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def unified_objective(trial, X_train, y_train):
    """Unified objective function for Optuna hyperparameter optimization."""
    model_type = trial.suggest_categorical("model_type", ["RandomForest", "XGBoost", "SVR", "KNN"])
    
    if model_type == "RandomForest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "bootstrap": trial.suggest_categorical('bootstrap', [True, False]),
        }
        model = RandomForestRegressor(random_state=42, **params)
        
    elif model_type == "XGBoost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        }
        model = XGBRegressor(random_state=42, verbosity=0, **params)
        
    elif model_type == "SVR":
        params = {
            "C": trial.suggest_float("C", 0.1, 10),
            "epsilon": trial.suggest_float("epsilon", 0.01, 1.0),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly", "sigmoid"]),
            "gamma": trial.suggest_categorical('gamma', ['scale', 'auto'])
        }
        model = SVR(**params)
        
    elif model_type == "KNN":
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 30),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_int("p", 1, 2)
        }
        model = KNeighborsRegressor(**params)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Use negative RMSE as the objective (Optuna minimizes, so we negate)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_root_mean_squared_error')
    score = -scores.mean()  # Convert back to positive RMSE
    
    trial.set_user_attr("model_type", model_type)
    return score

def get_model_targets():
    """Define target columns for each model."""
    return {
        "surface_water_area_in_swb": {
            'target_column': 'target_norm_surface_water_area_in_swb_kharif',
            'norm_factor_column': 'norm_factor_surface_water_area_in_swb_kharif',
            'id_columns': ['uid', 'year']
        },
        "area_under_hiz": {
            'target_column': 'target_norm_area_wb_hiz_kharif',
            'norm_factor_column': 'norm_factor_area_wb_hiz_kharif',
            'id_columns': ['uid', 'year']
        },
        "area_under_liz": {
            'target_column': 'target_norm_area_wb_liz_kharif',
            'norm_factor_column': 'norm_factor_area_wb_liz_kharif',
            'id_columns': ['uid', 'year']
        },
        "cropping_area_hiz": {
            'target_column': 'target_norm_cropping_area_hiz_kharif',
            'norm_factor_column': 'norm_factor_cropping_area_hiz_kharif',
            'id_columns': ['uid', 'year']
        },
        "crop_health_hiz": {
            'target_column': 'target_crop_health_hiz_kharif',
            'norm_factor_column': 'norm_factor_crop_health_hiz_kharif',
            'id_columns': ['uid', 'year']
        },
        "cropping_area_liz": {
            'target_column': 'target_norm_cropping_area_liz_kharif',
            'norm_factor_column': 'norm_factor_cropping_area_liz_kharif',
            'id_columns': ['uid', 'year']
        },
        "cropping_area_other": {
            'target_column': 'target_norm_cropping_area_other_kharif',
            'norm_factor_column': 'norm_factor_cropping_area_other_kharif',
            'id_columns': ['uid', 'year']
        },
        "crop_health_liz": {
            'target_column': 'target_crop_health_liz_kharif',
            'norm_factor_column': 'norm_factor_crop_health_liz_kharif',
            'id_columns': ['uid', 'year']
        },
        "crop_health_other": {
            'target_column': 'target_crop_health_other_kharif',
            'norm_factor_column': 'norm_factor_crop_health_other_kharif',
            'id_columns': ['uid', 'year']
        },
        "abs_gw_liz": {
            'target_column': 'target_norm_gw_irrigation_liz_kharif',
            'norm_factor_column': 'norm_factor_gw_irrigation_liz_kharif',
            'id_columns': ['uid', 'year']
        },
        "abs_gw_other": {
            'target_column': 'target_norm_gw_irrigation_other_kharif',
            'norm_factor_column': 'norm_factor_gw_irrigation_other_kharif',
            'id_columns': ['uid', 'year']
        },
        "mws_soge": {
            'target_column': 'target_mws_soge_value',
            'norm_factor_column': 'norm_factor_mws_soge_value',
            'id_columns': ['uid', 'year']
        }
    }

def train_model(model_name, df_train, df_test, df_target_train, df_target_test):
    """Train a model with hyperparameter optimization."""
    logger.info(f"=== Starting tuning for {model_name} ===")
    
    # Get target configuration
    target_config = get_model_targets().get(model_name)
    if not target_config:
        logger.warning(f"No target configuration found for {model_name}. Skipping.")
        return
    
    target_col = target_config["target_column"]
    norm_factor_col = target_config["norm_factor_column"]
    
    # Check if target column exists
    if target_col not in df_target_train.columns or norm_factor_col not in df_target_train.columns:
        logger.warning(f"Column not found in target dataframe for {model_name}. Skipping.")
        return
    
    feature_cols = [col for col in df_train.columns if col not in ["uid"]]
    
    X_train = df_train[feature_cols]
    y_train = df_target_train[target_col]
    norm_factor_train = df_target_train[norm_factor_col]
    X_test = df_test[feature_cols]
    y_test = df_target_test[target_col]
    norm_factor_test = df_target_test[norm_factor_col]
    
    # Check for missing values
    if X_train.isnull().any().any():
        logger.warning(f"Missing values found in features for {model_name}. Consider handling them.")
    
    if y_train.isnull().any():
        logger.warning(f"Missing values found in target for {model_name}. Consider handling them.")
    
    # Run hyperparameter optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: unified_objective(trial, X_train, y_train), n_trials=50)
    
    best_params = study.best_params
    best_model_type = study.best_trial.user_attrs["model_type"]
    
    logger.info(f"Best model for {model_name}: {best_model_type} with RMSE = {study.best_value:.4f}")
    
    # Retrain best model on full train set
    model_params = {k: v for k, v in best_params.items() if k != "model_type"}
    
    if best_model_type == "RandomForest":
        model = RandomForestRegressor(random_state=42, **model_params)
    elif best_model_type == "XGBoost":
        model = XGBRegressor(random_state=42, verbosity=0, **model_params)
    elif best_model_type == "SVR":
        model = SVR(**model_params)
    elif best_model_type == "KNN":
        model = KNeighborsRegressor(**model_params)
    else:
        raise ValueError(f"Unknown best model type: {best_model_type}")
    
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    preds = model.predict(X_test)
    y_preds = preds*norm_factor_test
    y_test *= norm_factor_test
    rmse = root_mean_squared_error(y_test, y_preds)
    r2 = r2_score(y_test, y_preds)
    
    logger.info(f"[FINAL] {model_name} | Model: {best_model_type} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
    
    # Save model and config
    model_filename = f"{model_name}_{best_model_type}.pkl"
    params_filename = f"{model_name}_{best_model_type}_params.json"
    
    joblib.dump(model, os.path.join(MODEL_DIR, model_filename))
    
    # Save parameters along with model type and performance metrics
    save_params = {
        "model_type": best_model_type,
        "parameters": model_params,
        "performance": {
            "rmse": rmse,
            "r2": r2
        },
        "target_config": target_config
    }
    
    with open(os.path.join(TUNING_DIR, params_filename), "w") as f:
        json.dump(save_params, f, indent=4)
    
    logger.info(f"Saved model: {model_filename}")
    logger.info(f"Saved parameters: {params_filename}")

def main():
    """Main training function."""
    logger.info("Starting unified Optuna model training...")
    
    # Get list of available feature files
    if not os.path.exists(TRAIN_FEATURES_DIR):
        logger.error(f"Training features directory not found: {TRAIN_FEATURES_DIR}")
        return
    
    feature_files = [f for f in os.listdir(TRAIN_FEATURES_DIR) if f.endswith(".csv")]
    
    if not feature_files:
        logger.error("No CSV files found in training features directory")
        return
    
    logger.info(f"Found {len(feature_files)} feature files to process")
    
    for file in feature_files:
        model_name = file.replace(".csv", "")
        logger.info(f"Processing model: {model_name}")
        
        # Paths for features
        path_train_features = os.path.join(TRAIN_FEATURES_DIR, file)
        path_test_features = os.path.join(TEST_FEATURES_DIR, file)
        
        # Paths for targets
        path_train_targets = os.path.join(TRAIN_TARGETS_DIR, file)
        path_test_targets = os.path.join(TEST_TARGETS_DIR, file)
        
        # Check if all required files exist
        missing_files = []
        if not os.path.exists(path_train_features):
            missing_files.append(path_train_features)
        if not os.path.exists(path_test_features):
            missing_files.append(path_test_features)
        if not os.path.exists(path_train_targets):
            missing_files.append(path_train_targets)
        if not os.path.exists(path_test_targets):
            missing_files.append(path_test_targets)
        
        if missing_files:
            logger.warning(f"Missing files for {model_name}: {missing_files}. Skipping.")
            continue
        
        try:
            # Load dataframes
            df_train = pd.read_csv(path_train_features)
            df_test = pd.read_csv(path_test_features)
            df_target_train = pd.read_csv(path_train_targets)
            df_target_test = pd.read_csv(path_test_targets)
            
            
            logger.info(f"Loaded data for {model_name}:")
            logger.info(f"  Train features: {df_train.shape}")
            logger.info(f"  Test features: {df_test.shape}")
            logger.info(f"  Train targets: {df_target_train.shape}")
            logger.info(f"  Test targets: {df_target_test.shape}")
            
            # Train model
            train_model(model_name, df_train, df_test, df_target_train, df_target_test)
            
        except Exception as e:
            logger.error(f"Error processing {model_name}: {str(e)}")
            continue
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()