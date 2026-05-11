# MLOps Assignment: Predictive Maintenance Classification

## Project Overview
This project focuses on building a full MLOps pipeline for predictive maintenance, specifically a multi-class failure type prediction model for heavy-equipment manufacturing. The goal is to detect and classify machine failures (Tool Wear, Heat Dissipation, Power, Overstrain) using sensor readings to minimize costly downtime.

### Key Tasks:
- **Data Loading & Validation:** Ingesting sensor data and validating its schema using Pandera.
- **Exploratory Data Analysis (EDA):** Understanding data distributions, feature engineering, and identifying imbalances.
- **Experiment Tracking & Model Selection:** Training and evaluating multiple models (Logistic Regression, Random Forest, XGBoost, LightGBM) using MLflow, followed by hyperparameter tuning with Optuna.
- **Drift Detection & Monitoring:** Using Evidently to detect data drift between production batches and the training data.
- **Explainability & Insights:** Applying SHAP to understand model predictions and identify key drivers for each failure type.

## Data Description
The dataset consists of sensor readings (Air temperature, Process temperature, Rotational speed, Torque, Tool wear) and a `Failure_Type` target variable. Derived features like `Power_W` (Mechanical Power) and `Temp_diff` (Temperature Differential) were engineered.

**Failure Classes:**
| Code | Name            | Description                 |
|------|-----------------|-----------------------------|
| 0    | No Failure      | Machine operating normally  |
| 1    | TWF             | Tool Wear Failure           |
| 2    | HDF             | Heat Dissipation Failure    |
| 3    | PWF             | Power Failure               |
| 4    | OSF             | Overstrain Failure          |

## Key Findings and Insights

### 1. Data Imbalance & SMOTE
- The `Failure_Type` distribution is highly imbalanced, with 'No Failure' being the dominant class. SMOTE was applied to the training set to address this, preventing data leakage into the validation set.

### 2. Model Performance & Selection
- **Tuned XGBoost** emerged as the best model with a **Macro F1 score of 0.7604** on the validation set, showing a 0.0124 improvement over the baseline XGBoost. Macro F1 was chosen over accuracy due to the imbalanced nature of the dataset, as it provides a more robust measure of performance across all failure classes, which is crucial for minimizing operational costs associated with missed failure predictions.

### 3. The TWF Problem: Data Scarcity
- Despite SMOTE and hyperparameter tuning, the **Tool Wear Failure (TWF)** class (Class 1) consistently showed an F1 score of 0.0. This is attributed to **data scarcity**, as the original training data contained only 30 real instances of TWF. A critical next step is to collect more real-world TWF data.

### 4. Data Drift in Stress Conditions
- The `current` batch showed no data drift, indicating stable operation. However, the `stress` batch exhibited significant data drift in `Rotational speed` (decreased), `Torque` (increased), and most notably, `Tool wear` (significantly increased). This suggests that machines under 'stress' operate with higher loads and accelerated tool degradation.
- **Implication:** The model should be retrained with data reflecting these stress conditions to maintain predictive accuracy. The maintenance schedule needs dynamic adjustment based on these drifted parameters to prevent increased risks of TWF and OSF.

### 5. SHAP Explainability & Actionable Recommendations
SHAP analysis provided crucial insights into feature importance for each failure type:
- **TWF (Tool Wear Failure):** Top driver is `Tool wear`. As tool wear accumulates, the risk of TWF increases.
- **HDF (Heat Dissipation Failure):** Top driver is `Temp_diff` (Process temperature - Air temperature). A large temperature differential indicates overheating.
- **PWF (Power Failure):** Top driver is `Power_W` (derived from Torque and Rotational speed). Issues in mechanical power output are key.
- **OSF (Overstrain Failure):** Top driver is `Tool wear`. High tool wear can lead to increased resistance and overstrain.

**Actionable Recommendation:**
- **Condition:** When `Tool wear` is detected to be significantly increasing (e.g., 20% above the historical average from `train` data, or a rapid rate of change is observed).
- **Risked failure class:** Tool Wear Failure (TWF) and Overstrain Failure (OSF).
- **Action:** Implement real-time monitoring of `Tool wear` and establish an adaptive maintenance protocol that triggers preemptive tool replacement or detailed machine inspection when wear thresholds are exceeded. This proactive approach will mitigate the risk of catastrophic failures and reduce unplanned downtime.

## Getting Started

### Dependencies
Key libraries used in this project:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `imblearn` (for SMOTE)
- `mlflow`
- `optuna`
- `evidently`
- `shap`

Install these using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imblearn mlflow optuna evidently shap
Data Files
Ensure the following data files are present in a data/ directory relative to the notebook:

train.csv
current.csv
stress.csv
Running the Notebook
Clone this repository.
Install the dependencies.
Open and run the Predictive_Maintenance_Classification.ipynb notebook in a Jupyter-compatible environment (e.g., Google Colab, Jupyter Lab).
