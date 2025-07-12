import pandas as pd
import numpy as np
import os
import logging

# Creates a log directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('transform_flatten_raw_data.py')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "transform_flatten_raw_data.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded data from {path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.exception(f"Failed to load data from {path}")
        raise

def flatten_raw_data(df: pd.DataFrame, season: str) -> pd.DataFrame:
    season = season.lower()
    if season not in ["kharif", "rabi", "zaid"]:
        raise ValueError(f"Unsupported season: {season}. Choose from 'kharif', 'rabi', or 'zaid'.")

    rows = []
    years = ['2017', '2018', '2019', '2020', '2021', '2022']

    for _, row in df.iterrows():
        uid = row['uid']
        for year in years:
            y = str(year)
            new_row = {
                'uid': uid,
                'year': int(y),
                'area_in_ha': row.get(f'area_wb_hiz_{season}_{y}', 0) +
                              row.get(f'area_wb_liz_{season}_{y}', 0) +
                              row.get(f'area_wb_other_{season}_{y}', 0),
                f'rainfall_{season}': row.get(f'rainfall_{season}_{y}', np.nan),
                f'runoff_{season}': row.get(f'runoff_{season}_{y}', np.nan),
                f'max_sw_area': row.get(f'max_sw_area_{y}', np.nan),
                'dryspell_length': row.get(f'dryspell_length_{y}', np.nan),
                'rainfall_deviation_class': row.get(f'rainfall_deviation_class_{y}', None),
                'spi_class': row.get(f'spi_class_{y}', None),
                'number_of_weeks_in_mild_drought': row.get(f'number_of_weeks_in_mild_drought_{y}', np.nan),
                'number_of_weeks_in_moderate_drought': row.get(f'number_of_weeks_in_moderate_drought_{y}', np.nan),
                'number_of_weeks_in_severe_drought': row.get(f'number_of_weeks_in_severe_drought_{y}', np.nan),
                'is_drought_year': row.get(f'is_drought_{y}', False),
                f'surface_water_area_in_swb_{season}': row.get(f'{season}_water_{y}', np.nan),
                f'area_wb_hiz_{season}': row.get(f'area_wb_hiz_{season}_{y}', np.nan),
                f'area_wb_liz_{season}': row.get(f'area_wb_liz_{season}_{y}', np.nan),
                f'area_wb_other_{season}': row.get(f'area_wb_other_{season}_{y}', np.nan),
                f'max_area_hiz': row.get(f'max_area_hiz_{y}', np.nan),
                f'max_cropping_area_hiz': row.get(f'max_cropping_area_hiz_{y}', np.nan),
                f'max_area_liz': row.get(f'max_area_liz_{y}', np.nan),
                f'max_cropping_area_liz': row.get(f'max_cropping_area_liz_{y}', np.nan),
                f'max_area_other': row.get(f'max_area_other_{y}', np.nan),
                f'max_cropping_area_other': row.get(f'max_cropping_area_other_{y}', np.nan),
                f'max_cropping_area': row.get(f'max_cropping_area_hiz_{y}', np.nan) + row.get(f'max_cropping_area_liz_{y}', np.nan) + row.get(f'max_cropping_area_other_{y}', np.nan),
                f'monsoon_onset_dev_days_{season}': row.get(f'monsoon_onset_dev_days_{y}', np.nan),
                f'cropping_area_hiz_{season}': row.get(f'cropping_area_hiz_{season}_{y}', np.nan),
                f'cropping_area_liz_{season}': row.get(f'cropping_area_liz_{season}_{y}', np.nan),
                f'cropping_area_other_{season}': row.get(f'cropping_area_other_{season}_{y}', np.nan),
                f'crop_health_hiz_{season}': row.get(f'{season}_high_impact_ndvi_{y}', np.nan),
                f'crop_health_liz_{season}': row.get(f'{season}_low_impact_ndvi_{y}', np.nan),
                f'crop_health_other_{season}': row.get(f'{season}_other_ndvi_{y}', np.nan),
                f'temp_extreme_intensity_{season}': row.get(f'{season}_intensity_{y}', np.nan),
                f'temp_extreme_days_proportion_{season}': row.get(f'{season}_frequency_{y}', np.nan),
                f'gw_irrigation_other_{season}': row.get(f'gw_abstracted_other_{season}_{y}', np.nan),
                f'gw_irrigation_liz_{season}': row.get(f'gw_abstracted_liz_{season}_{y}', np.nan),
                'total_gw_irrigation': row.get(f'gw_abstracted_oz_liz_{y}', np.nan),
                'mws_soge_value': row.get(f'mws_soge_value_{y}', np.nan),
                
                # Add 2017 baseline features - created from raw data
                f'rainfall_{season}_2017': row.get(f'rainfall_{season}_2017', np.nan),
                f'runoff_{season}_2017': row.get(f'runoff_{season}_2017', np.nan),
                f'dryspell_length_2017': row.get(f'dryspell_length_2017', np.nan),
                f'rainfall_deviation_class_2017': row.get(f'rainfall_deviation_class_2017', None),
                f'spi_class_2017': row.get(f'spi_class_2017', None),
                f'monsoon_onset_dev_days_2017': row.get(f'monsoon_onset_dev_days_2017', np.nan),
                f'max_sw_area_2017': row.get('max_sw_area_2017', np.nan),
                f'surface_water_area_in_swb_{season}_2017': row.get(f'{season}_water_2017', np.nan),
                f'temp_extreme_intensity_{season}_2017': row.get(f'{season}_intensity_2017', np.nan),
                f'temp_extreme_days_proportion_{season}_2017': row.get(f'{season}_frequency_2017', np.nan),
                f'area_wb_other_{season}_2017': row.get(f'area_wb_other_{season}_2017', np.nan),
                f'cropping_area_other_{season}_2017': row.get(f'cropping_area_other_{season}_2017', np.nan),
                f'max_cropping_area_other_2017': row.get(f'max_cropping_area_other_2017', np.nan),
                f'crop_health_hiz_{season}_2017': row.get(f'{season}_high_impact_ndvi_2017', np.nan),
                f'crop_health_liz_{season}_2017': row.get(f'{season}_low_impact_ndvi_2017', np.nan),
                f'crop_health_other_{season}_2017': row.get(f'{season}_other_ndvi_2017', np.nan),
            }

            for col in ['SO_1', 'SO_2', 'SO_3', 'SO_4', 'SO_5', 'SO_6', 'SO_7', 'SO_8', 'SO_9', 'SO_10', 'SO_11', 'isAlluviumAquifer']:
                new_row[col] = row.get(col, np.nan)

            rows.append(new_row)

    result_df = pd.DataFrame(rows)

    result_df[f"total_cropping_area_{season}"] = (
        result_df[f"cropping_area_hiz_{season}"] +
        result_df[f"cropping_area_liz_{season}"] +
        result_df[f"cropping_area_other_{season}"]
    )

    result_df[f"max_cropping_area"] = (
        result_df
        .sort_values(["uid", "year"])
        .groupby("uid")[f"total_cropping_area_{season}"]
        .cummax()
    )

    # lag_cols = [col for col in result_df.columns if col.endswith(f"_{season}")]
    lag_cols = [
        f'max_sw_area',
        f'surface_water_area_in_swb_{season}',
        f'area_wb_hiz_{season}',
        f'area_wb_liz_{season}',
        f'area_wb_other_{season}',
        f'max_area_hiz',
        f'max_cropping_area_hiz',
        f'max_area_liz',
        f'max_cropping_area_liz',
        f'max_area_other',
        f'max_cropping_area_other',
        f'max_cropping_area',
        f'cropping_area_hiz_{season}',
        f'cropping_area_liz_{season}',
        f'cropping_area_other_{season}',
        f'crop_health_hiz_{season}',
        f'crop_health_liz_{season}',
        f'crop_health_other_{season}',
        f'gw_irrigation_other_{season}',
        f'gw_irrigation_liz_{season}',
        'total_gw_irrigation',
        'mws_soge_value'
    ]
    result_df.sort_values(['uid', 'year'], inplace=True)

    for col in lag_cols:
        result_df[f"{col}_prev"] = result_df.groupby("uid")[col].shift(1)

    result_df.dropna(inplace=True)
    result_df.reset_index(drop=True, inplace=True)

    logger.info(f"Final flattened {season} dataset shape: {result_df.shape}")
    return result_df


def save_data(data: pd.DataFrame, data_path: str, season: str) -> None:
    """Save flattened dataset"""
    try:
        interim_data_path = os.path.join(data_path, "interim")
        os.makedirs(interim_data_path, exist_ok=True)
        filename = f"flattened_dataset_{season.lower()}.csv"
        data.to_csv(os.path.join(interim_data_path, filename), index=False)
        logger.debug("Flattened Dataset (%s) saved to %s", season, interim_data_path)
    except Exception as e:
        logger.error("Error while saving the data for season %s: %s", season, e)
        raise

def main():
    try:
        raw_data_path = "./data/raw/final_merged_dataset.csv"
        df = load_data(raw_data_path)
        for season in ["kharif", "rabi", "zaid"]:
            flattened = flatten_raw_data(df, season=season)
            save_data(flattened, './data', season=season)
    except Exception:
        logger.exception("Failed in transform_flatten_raw_data pipeline")

if __name__ == "__main__":
    main()
