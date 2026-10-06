# AgriPredict: Smart Farm Yield Analytics and Prediction

**AgriPredict** is a Python-based agricultural analytics and machine learning system that predicts crop yield using weather, soil, and farm-level variables. The application combines a validated machine learning pipeline with a polished Tkinter dashboard for prediction, dynamic yield gauges, regional profile benchmarking, what-if scenario simulations, dataset exploration, model comparison, and feature-importance analysis.

## Project Goal

Estimate crop yield in tonnes per hectare ($\text{t/ha}$) using inputs such as seasonal rainfall, mean temperature, sunlight hours, soil pH, organic carbon, nitrogen application, irrigation status, and farm size.

## Project Structure

- `crop_yield_core.py` — data loading, validation, cleaning, model training, feature importance, and prediction logic
- `generate_data.py` — synthetic data generation for the farm-yield dataset
- `gui.py` — AgriPredict dashboard UI for prediction and analytics
- `run_pipeline.py` — end-to-end pipeline runner
- `data/crop_yield_data.csv` — dataset used by the ML workflow
- `crop_yield_results/` — model metrics, importance values, and saved artifacts
- `requirements.txt` — project dependencies

## Dataset

The project uses a farm-yield dataset with 2,988 cleaned rows after removing duplicates and implausible values. Key features include:

- `rainfall_mm` (Seasonal precipitation, mm)
- `avg_temp_c` (Mean temperature, °C)
- `sunlight_hours` (Sunshine duration, hrs/day)
- `soil_ph` (Soil pH scale)
- `organic_carbon_pct` (Soil organic carbon, %)
- `nitrogen_kg_ha` (Nitrogen applied, kg/ha)
- `irrigated` (Irrigation status: 0 = No, 1 = Yes)
- `farm_size_ha` (Farm landholding size, ha)
- `yield_t_ha` (**Target**: Crop yield, tonnes/hectare)

## ML Workflow

The pipeline performs:

1. Data loading and schema validation
2. Duplicate and implausible row removal
3. Missing value handling via median imputation inside cross-validation pipelines
4. 80/20 train/test split
5. 5-fold cross-validation model comparison
6. Model selection using CV MAE
7. Permutation importance evaluation
8. Artifact serialization for the AgriPredict dashboard

## Models Compared

The project benchmarks the following models:

- Mean baseline
- Linear regression
- Polynomial ridge (degree 2)
- Random forest
- Gradient boosting (`HistGradientBoostingRegressor`)

## Current Best Model

The model selected by the project is: `Gradient boosting`

Model metrics from the training pipeline:

- **CV MAE**: 0.640 t/ha
- **CV R²**: 0.700
- **Test MAE**: 0.641 t/ha
- **Test RMSE**: 0.852 t/ha
- **Test R²**: 0.708

Permutation importance (increase in test-set MAE when shuffled):

1. `rainfall_mm`: +0.383 t/ha
2. `soil_ph`: +0.339 t/ha
3. `avg_temp_c`: +0.284 t/ha
4. `nitrogen_kg_ha`: +0.242 t/ha
5. `irrigated`: +0.222 t/ha
6. `sunlight_hours`: +0.079 t/ha
7. `organic_carbon_pct`: +0.016 t/ha
8. `farm_size_ha`: +0.006 t/ha

## Setup

1. Clone the repository.
2. Create a virtual environment (recommended):

```bash
python -m venv .venv
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

### 1) Run the ML pipeline

```bash
python run_pipeline.py
```

This trains the models, evaluates cross-validation scores, and saves artifacts to `crop_yield_results/`.

### 2) Launch the AgriPredict GUI dashboard

```bash
python gui.py
```

The AgriPredict desktop app includes pages for:

- **Overview**: KPIs, system status, and dataset explorer
- **Yield Prediction**: Dynamic gauge, farm input profile benchmark, what-if scenario comparison, and prediction history trend
- **Data Insights**: Bivariate ground-truth scatter and box plots
- **Model Performance**: Cross-validation & test metric comparisons
- **Feature Importance**: Permutation importance error penalties
- **About Project**: Methodology and architecture details

## GUI Features

- Professional agriculture-themed dashboard (**AgriPredict**)
- Real-time yield prediction with input validation
- Dynamic semicircular yield gauge
- Farm input vs. regional dataset profile benchmarking
- Interactive What-If sensitivity simulator with yield impact delta
- Session prediction history table and trend sequence chart
- Model comparison table and error metrics
- Permutation feature importance visualization
- Dataset preview modal with search filtering
- About AgriPredict project information page

## Notes

- The project preserves the verified ML logic and uses the trained Gradient Boosting model for all predictions.
- All displayed metrics come from the actual project dataset and pipeline outputs.

## Requirements

Python 3.10+ recommended. Dependencies include:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `joblib`
- `jupyter`
- `ipykernel`

## License

This project is intended for academic and demonstration use.
