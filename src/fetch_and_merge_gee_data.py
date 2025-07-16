import pandas as pd
import numpy as np
import os
import logging
import yaml
import ee

# Creates a log directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('fetch_and_merge_gee_data.py')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "fetch_and_merge_gee_data.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def initialize_ee(project_id):
    """Authenticate and initialize Earth Engine with a service account."""
    ee.Initialize(project=project_id)

def load_gee_data(data_url: str) -> pd.DataFrame:
    """
    Load data from a Google Earth Engine FeatureCollection asset into a DataFrame.

    Args:
        data_url (str): GEE asset URL.

    Returns:
        pd.DataFrame: DataFrame containing the 'properties' of the features.
    """
    try:
        feature_collection = ee.FeatureCollection(data_url)
        features = feature_collection.getInfo()['features']
        if not features:
            logger.warning("No features found in GEE asset: %s", data_url)
            return pd.DataFrame()

        df = pd.DataFrame([f['properties'] for f in features])
        logger.debug("Loaded %d rows from GEE asset: %s", len(df), data_url)
        return df

    except Exception as e:
        logger.exception("Error loading GEE data from %s: %s", data_url, e)
        raise


def preprocess_rainfall_data(url: str) -> pd.DataFrame:
    """
    Preprocess rainfall data by removing unnecessary columns.

    Args:
        url (str): GEE asset URL for rainfall data.

    Returns:
        pd.DataFrame: Cleaned rainfall DataFrame.
    """
    df = load_gee_data(url)
    drop_cols = ['id', 'area_in_ha', 'DN']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    logger.info("Preprocessed rainfall data. Final shape: %s", df.shape)
    return df

def preprocess_runoff_data(url):
    df = load_gee_data(url)
    drop_cols = [col for col in ['id', 'area_in_ha', 'DN'] if col in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    logger.info("Preprocessed runoff data. Final shape: %s", df.shape)
    return df


def preprocess_max_surface_water_area_swb(url):
    df = load_gee_data(url)
    drop_cols = [col for col in ['id', 'area_in_ha', 'DN'] if col in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    logger.info("Preprocessed max surface water area in swb data. Final shape: %s", df.shape)
    return df


def preprocess_stream_order_areas_data(url: str) -> pd.DataFrame:
    """
    Preprocess stream order areas data by flattening keys '1' to '11' into columns SO_1 to SO_11.

    Args:
        url (str): GEE asset URL for stream order areas.

    Returns:
        pd.DataFrame: DataFrame with columns ['uid', 'SO_1', ..., 'SO_11'].
    """
    df = load_gee_data(url)
    drop_cols = [col for col in ['id', 'area_in_ha', 'DN'] if col in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    if df.empty:
        logger.warning("Stream order data is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    output_rows = []
    stream_order_keys = [str(i) for i in range(1, 12)]

    for _, row in df.iterrows():
        data = {'uid': row.get('uid')}
        for k in stream_order_keys:
            data[f'SO_{k}'] = np.round(row.get(k, 0.0), 3)
        output_rows.append(data)

    processed_df = pd.DataFrame(output_rows).fillna(0)

    logger.info("Preprocessed stream order data. Final shape: %s", processed_df.shape)
    return processed_df


def categorize_spi(spi: float) -> str:
    """
    Categorize SPI (Standardized Precipitation Index) values into drought/wetness levels.
    """
    spi = np.round(spi, 2)
    if spi < -2.0: return "Extremely Dry"
    elif -2.0 <= spi <= -1.5: return "Severely Dry"
    elif -1.5 < spi <= -1.0: return "Moderately Dry"
    elif -1.0 < spi <= 0: return "Mildly Dry"
    elif 0 < spi <= 1.0: return "Mildly Wet"
    elif 1.0 < spi <= 1.5: return "Moderately Wet"
    elif 1.5 < spi <= 2.0: return "Severely Wet"
    elif spi > 2.0: return "Extremely Wet"
    return "UNK"


def categorize_rainfall_deviation(p: float) -> str:
    """
    Categorize rainfall deviation percentages into standard IMD-based classes.
    """
    if p >= 60: return 'Large Excess'
    elif 20 <= p < 60: return 'Excess'
    elif -19 <= p < 20: return 'Normal'
    elif -59 <= p < -19: return 'Deficient'
    elif -99 <= p < -59: return 'Large Deficient'
    elif p < -99: return 'No Rain'
    return 'UNK'


def preprocess_drought_data(prefix_url: str) -> pd.DataFrame:
    """
    Preprocess drought-related features across years from Earth Engine assets.

    Args:
        prefix_url (str): Base URL of GEE FeatureCollection per year, e.g., '.../drought_2017'

    Returns:
        pd.DataFrame: Combined drought metrics across years.
    """
    all_years_df = pd.DataFrame()

    for year in range(2017, 2023):
        try:
            logger.info(f"Processing drought data for year {year}")
            url = f"{prefix_url}_{year}"
            feature_collection = ee.FeatureCollection(url)
            features = feature_collection.getInfo()['features']
            df = pd.DataFrame([f['properties'] for f in features])

            # Mean rainfall deviation
            rain_dev_cols = [col for col in df.columns if col.startswith("monthly_rainfall_deviation")]
            df['mean_monthly_rainfall_deviation'] = df[rain_dev_cols].mean(axis=1)
            df[f'rainfall_deviation_class_{year}'] = df['mean_monthly_rainfall_deviation'].apply(categorize_rainfall_deviation)

            # Mean SPI
            spi_cols = [col for col in df.columns if col.startswith("spi_")]
            df['mean_spi_value'] = df[spi_cols].mean(axis=1)
            df[f'spi_class_{year}'] = df['mean_spi_value'].apply(categorize_spi)

            # Handle drought weeks
            mod_col = f'number_of_weeks_in_moderate_drought_{year}'
            sev_col = f'number_of_weeks_in_severe_drought_{year}'
            df[mod_col] = df.get(mod_col, 0).fillna(0)
            df[sev_col] = df.get(sev_col, 0).fillna(0)
            df[f'is_drought_{year}'] = (df[mod_col] + df[sev_col]) >= 5

            # Filter invalid entries
            df = df[df["mean_monthly_rainfall_deviation"] != -9999]

            # Select and rename final columns
            selected_cols = [
                'uid',
                f'dryspell_length_{year}',
                f'rainfall_deviation_class_{year}',
                f'spi_class_{year}',
                f'number_of_weeks_in_mild_drought_{year}',
                mod_col,
                sev_col,
                f'is_drought_{year}'
            ]
            df_year = df[selected_cols]

            # Merge yearly results
            if all_years_df.empty:
                all_years_df = df_year
            else:
                all_years_df = pd.merge(all_years_df, df_year, on='uid', how='outer')

            logger.info(f"Drought data for {year} processed. Shape: {df_year.shape}")

        except Exception as e:
            logger.warning(f"Failed to process drought data for year {year}: {e}")

    logger.info(f"Total drought data shape after merge: {all_years_df.shape}")
    return all_years_df

def preprocess_surface_water_area_swb(url: str) -> pd.DataFrame:
    """
    Preprocess surface water area data by dropping irrelevant columns.
    """
    df = load_gee_data(url)
    drop_cols = [col for col in ['id', 'area_in_ha', 'DN'] if col in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    logger.info("Preprocessed surface water area SWB data. Shape: %s", df.shape)
    return df


def preprocess_total_area_zonewise_by_wb_creation(url: str) -> pd.DataFrame:
    """
    Load total area zonewise by waterbody creation. No column drop needed.
    """
    df = load_gee_data(url)
    drop_cols = [col for col in ['id', 'area_in_ha', 'DN'] if col in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    logger.info("Loaded total area zonewise by WB creation. Shape: %s", df.shape)
    return df


def preprocess_monsoon_onset(url: str) -> pd.DataFrame:
    """
    Load and filter monsoon onset data to retain only specific year columns.
    """
    df = load_gee_data(url)
    drop_cols = [col for col in ['id', 'area_in_ha', 'DN'] if col in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    required_cols = [
        'uid',
        'monsoon_onset_dev_days_2017',
        'monsoon_onset_dev_days_2018',
        'monsoon_onset_dev_days_2019',
        'monsoon_onset_dev_days_2020',
        'monsoon_onset_dev_days_2021',
        'monsoon_onset_dev_days_2022'
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logger.warning("Missing monsoon onset columns: %s", missing)
    df = df[[col for col in required_cols if col in df.columns]]
    logger.info("Preprocessed monsoon onset data. Shape: %s", df.shape)
    return df


def preprocess_cropping_areas_zonewise(url: str) -> pd.DataFrame:
    """
    Preprocess cropping area data zonewise by dropping irrelevant columns.
    """
    df = load_gee_data(url)
    drop_cols = [col for col in ['id', 'area_in_ha', 'DN'] if col in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    logger.info("Preprocessed cropping areas zonewise data. Shape: %s", df.shape)
    return df


def preprocess_max_area_zonewise_by_wb_creation(url: str) -> pd.DataFrame:
    """
    Preprocess maximum area zonewise data by dropping irrelevant columns.
    """
    df = load_gee_data(url)
    drop_cols = [col for col in ['id', 'area_in_ha', 'DN'] if col in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    logger.info("Preprocessed max area zonewise by WB creation. Shape: %s", df.shape)
    return df

def preprocess_max_cropping_area_zonewise(url: str) -> pd.DataFrame:
    """
    Preprocess max cropping area data zonewise by dropping unnecessary columns.
    """
    df = load_gee_data(url)
    drop_cols = [col for col in ['id', 'area_in_ha', 'DN'] if col in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    logger.info("Preprocessed max cropping area zonewise data. Shape: %s", df.shape)
    return df


def preprocess_crop_health_zonewise(url: str) -> pd.DataFrame:
    """
    Preprocess crop health data zonewise by dropping unnecessary columns.
    """
    df = load_gee_data(url)
    drop_cols = [col for col in ['id', 'area_in_ha', 'DN'] if col in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    logger.info("Preprocessed crop health zonewise data. Shape: %s", df.shape)
    return df


def preprocess_temperature_metrics(url: str) -> pd.DataFrame:
    """
    Preprocess temperature-related metrics data by dropping unnecessary columns.
    """
    df = load_gee_data(url)
    drop_cols = [col for col in ['id', 'area_in_ha', 'DN'] if col in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    logger.info("Preprocessed temperature metrics data. Shape: %s", df.shape)
    return df


def preprocess_gw_abs_other_zone_metrics(url: str) -> pd.DataFrame:
    """
    Preprocess groundwater abstraction metrics for OZ/LIZ zones by dropping unnecessary columns.
    """
    df = load_gee_data(url)
    drop_cols = [col for col in ['id', 'area_in_ha', 'DN'] if col in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    logger.info("Preprocessed groundwater abstraction (OZ/LIZ) metrics data. Shape: %s", df.shape)
    return df

def preprocess_total_gw_abs_liz_oz(url: str) -> pd.DataFrame:
    """
    Preprocess total groundwater abstraction data for OZ and LIZ zones.
    Keeps only yearly abstraction columns and UID.
    """
    df = load_gee_data(url)
    expected_cols = [
        'gw_abstracted_oz_liz_2017', 'gw_abstracted_oz_liz_2018',
        'gw_abstracted_oz_liz_2019', 'gw_abstracted_oz_liz_2020',
        'gw_abstracted_oz_liz_2021', 'gw_abstracted_oz_liz_2022',
        'uid'
    ]
    drop_cols = [col for col in ['id', 'area_in_ha', 'DN'] if col in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        logger.warning("Missing columns in GW abstraction data: %s", missing_cols)
    df = df[[col for col in expected_cols if col in df.columns]]
    logger.info("Preprocessed total groundwater abstraction (LIZ+OZ) data. Shape: %s", df.shape)
    return df


def preprocess_soge_mws(url: str) -> pd.DataFrame:
    """
    Preprocess SOGE (Sum of Groundwater Extraction) metrics for microwatersheds.
    Renames columns for consistency across years.
    """
    df = load_gee_data(url)
    drop_cols = [col for col in ['id', 'DN'] if col in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    original_cols = [
        'mws_soge_2017_value', 'mws_soge_2018_value',
        'mws_soge_2019_value', 'mws_soge_2020_value',
        'mws_soge_2021_value', 'mws_soge_2022_value',
        'uid', 'area_in_ha'
    ]
    missing_cols = [col for col in original_cols if col not in df.columns]
    if missing_cols:
        logger.warning("Missing SOGE columns: %s", missing_cols)

    df = df[[col for col in original_cols if col in df.columns]]
    rename_map = {
        f"mws_soge_{year}_value": f"mws_soge_value_{year}"
        for year in range(2017, 2023)
        if f"mws_soge_{year}_value" in df.columns
    }
    df.rename(columns=rename_map, inplace=True)
    logger.info("Preprocessed SOGE MWS data. Shape: %s", df.shape)
    return df




def merge_all_data(is_alluvium_aquifer: bool = False, **kwargs) -> pd.DataFrame:
    """
    Merge all preprocessed datasets into one DataFrame based on 'uid'.
    All URLs must be provided in kwargs with appropriate keys.
    """
    try:
        logger.info("Starting data merging pipeline...")

        # Load and preprocess individual datasets
        rainfall_df = preprocess_rainfall_data(kwargs['rainfall_url'])
        runoff_df = preprocess_runoff_data(kwargs['runoff_url'])
        max_surface_water_area_df = preprocess_max_surface_water_area_swb(kwargs['max_surface_water_area_url'])
        stream_order_areas_df = preprocess_stream_order_areas_data(kwargs['stream_order_areas_url'])
        drought_df = preprocess_drought_data(kwargs['drought_prefix_url'])
        surface_water_area_swb_df = preprocess_surface_water_area_swb(kwargs['surface_water_area_swb_url'])
        total_area_zonewise_by_wb_df = preprocess_total_area_zonewise_by_wb_creation(kwargs['total_area_zonewise_by_wb_creation_url'])
        monsoon_onset_df = preprocess_monsoon_onset(kwargs['monsoon_onset_url'])
        cropping_areas_zonewise_df = preprocess_cropping_areas_zonewise(kwargs['cropping_areas_zonewise_url'])
        crop_health_zonewise_df = preprocess_crop_health_zonewise(kwargs['crop_health_zonewise_url'])
        temperature_metrics_df = preprocess_temperature_metrics(kwargs['temperature_metrics_url'])
        abstracted_gw_oz_df = preprocess_gw_abs_other_zone_metrics(kwargs['abstracted_gw_oz_url'])
        abstracted_gw_liz_df = preprocess_gw_abs_other_zone_metrics(kwargs['abstracted_gw_liz_url'])
        total_abstracted_gw_liz_oz_df = preprocess_total_gw_abs_liz_oz(kwargs['total_abstracted_gw_liz_oz_url'])
        soge_mws_df = preprocess_soge_mws(kwargs['soge_mws_url'])
        max_cropping_area_zonewise_df = preprocess_max_cropping_area_zonewise(kwargs['max_cropping_area_url'])
        max_area_zonewise_df = preprocess_max_area_zonewise_by_wb_creation(kwargs['max_area_zonewise_url'])

        # List of all DataFrames to merge
        datasets = [
            runoff_df, max_surface_water_area_df, stream_order_areas_df, drought_df,
            surface_water_area_swb_df, total_area_zonewise_by_wb_df, monsoon_onset_df,
            cropping_areas_zonewise_df, crop_health_zonewise_df, temperature_metrics_df,
            abstracted_gw_oz_df, abstracted_gw_liz_df, total_abstracted_gw_liz_oz_df,
            soge_mws_df, max_cropping_area_zonewise_df, max_area_zonewise_df
        ]

        merged_df = rainfall_df.copy()
        for idx, df in enumerate(datasets):
            logger.debug("Merging dataset %d with shape %s", idx + 1, df.shape)
            merged_df = pd.merge(merged_df, df, on='uid', how='inner')

        merged_df['isAlluviumAquifer'] = is_alluvium_aquifer
        logger.info("Data merging completed. Final shape: %s", merged_df.shape)

        return merged_df

    except KeyError as ke:
        logger.error("Missing required input key: %s", ke)
        raise

    except Exception as e:
        logger.exception("Unexpected error during data merging")
        raise

def load_zone_data(zone_name: str, is_alluvium_aquifer: bool = False) -> pd.DataFrame:
    """
    Loads and merges all GEE datasets for a given zone using merge_all_data.
    """
    base_path = "projects/ee-siy237536/assets/system-modelling"
    shared_prefix = f"{base_path}"

    # Zone-specific asset mappings
    urls = {
        'rainfall_url': f"{shared_prefix}/rainfall/{zone_name}_yearly_rainfall",
        'runoff_url': f"{shared_prefix}/runoff/{zone_name}_yearly_runoff",
        'max_surface_water_area_url': f"{shared_prefix}/max_surface_water_area/{zone_name}_max_sw_area_yearly",
        'stream_order_areas_url': f"projects/ee-siy237536/assets/stream-order-areas/SO_areas_of_{zone_name}_mwses",
        'drought_prefix_url': f"{shared_prefix}/drought/{zone_name}_drought",
        'surface_water_area_swb_url': f"{shared_prefix}/surface_area_under_swb/{zone_name}_surface_water_area_2017-2022",
        'total_area_zonewise_by_wb_creation_url': f"{shared_prefix}/total_aoi_zonewise/{zone_name}_total_aoi_zonewise",
        'monsoon_onset_url': f"{shared_prefix}/monsoon-onset-and-dev/{zone_name}_monsoon_onsetv2_2003-2023",
        'cropping_areas_zonewise_url': f"{shared_prefix}/cropping_areas_zonewise/{zone_name}_yearly_cropping_areas_zonewise",
        'crop_health_zonewise_url': f"{shared_prefix}/crop_health/{zone_name}_mws_seasonal_ndvi_by_zone_2017_2022",
        'temperature_metrics_url': f"{shared_prefix}/temperature/{zone_name}_seasonal_extreme_temp_metrics",
        'abstracted_gw_oz_url': f"{shared_prefix}/groundwater_abstracted/{zone_name}_yearly_gw_abstracted_otherzone_1500m_LI",
        'abstracted_gw_liz_url': f"{shared_prefix}/groundwater_abstracted/{zone_name}_yearly_gw_abstracted_low_impact_zone_1500m_LI",
        'total_abstracted_gw_liz_oz_url': f"{shared_prefix}/groundwater_abstracted/{zone_name}_yearly_total_gw_abstracted_1500m_LI",
        'soge_mws_url': f"{shared_prefix}/soge_mws/{zone_name}_mws_soge_2017_2022",
        'max_area_zonewise_url': f"{shared_prefix}/max_zonewise_area/{zone_name}_max_zonewise_area",
        'max_cropping_area_url': f"{shared_prefix}/max_cropping_areas_zonewise/{zone_name}_max_cropping_areas_zonewise"
    }

    # Special cases (for naming inconsistency)
    if zone_name == 'masalia':
        urls['total_abstracted_gw_liz_oz_url'] += "_old"
        urls['soge_mws_url'] += "_old"
    elif zone_name == 'mandalgarh':
        urls['total_abstracted_gw_liz_oz_url'] += "_old"
        urls['soge_mws_url'] += "_old"
    elif zone_name == 'shahpur':
        urls['total_abstracted_gw_liz_oz_url'] += ""
    elif zone_name == 'rampur_maniharan':
        urls['total_abstracted_gw_liz_oz_url'] += ""

    try:
        logger.info(f"Processing zone: {zone_name}")
        zone_df = merge_all_data(is_alluvium_aquifer=is_alluvium_aquifer, **urls)
        zone_df['zone'] = zone_name  
        logger.info(f"{zone_name} loaded. Shape: {zone_df.shape}")
        return zone_df
    except Exception as e:
        logger.error("Failed to load zone %s: %s", zone_name, str(e))
        raise

def save_data(merged_data: pd.DataFrame, data_path: str) -> None:
    """Save merged dataset"""
    try:
        raw_data_path = os.path.join(data_path, "raw")
        os.makedirs(raw_data_path, exist_ok=True)
        
        merged_data.to_csv(os.path.join(raw_data_path, "final_merged_dataset.csv"), index=False)
        logger.debug("Merged dataset saved to %s", raw_data_path)
    except Exception as e:
        logger.error("Unexpected error occurred while saving the data: %s", e)
        raise


if __name__ == "__main__":
    
    output_file = "data/raw/final_merged_dataset.csv"

    if os.path.exists(output_file):
        logger.info(f"File '{output_file}' already exists. Skipping data ingestion.")
        exit(0)

    logger.info("Initializing Earth Engine with service account...")

    # Replace with your service account email and path to JSON key
    initialize_ee(project_id='ee-siy237536')

    logger.info("Earth Engine initialized.")
    logger.info("Starting data ingestion for all zones...")

    zones_info = {
        "masalia": False,
        "mandalgarh": False,
        "shahpur": True,
        "rampur_maniharan": True,
        "boipariguda": False,
        "sawali": False,
        "devdurga": False,
        "pindwara": False
    }

    all_zone_dfs = []

    for zone, is_alluvium in zones_info.items():
        try:
            logger.info(f"Loading data for zone: {zone}")
            df = load_zone_data(zone, is_alluvium_aquifer=is_alluvium)
            logger.info(f"Loaded data shape for {zone}: {df.shape}")
            all_zone_dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to load data for zone: {zone} — {str(e)}", exc_info=True)

    if all_zone_dfs:
        try:
            final_merged_df = pd.concat(all_zone_dfs, ignore_index=True)
            logger.info(f"Merged dataset shape: {final_merged_df.shape}")
            save_data(final_merged_df, data_path="./data")
        except Exception as e:
            logger.error(f"Failed to merge or save dataset — {str(e)}", exc_info=True)
    else:
        logger.warning("No zone dataframes were successfully loaded — skipping merge and save.")
