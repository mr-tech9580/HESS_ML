from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from .driving import generate_driving_cycle

from .generate_data import (
    create_demand_dataset,
    create_policy_dataset
)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "processed"

MODELS_DIR = ROOT / "models"

METRICS_DIR = ROOT / "results" / "metrics"


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

# This is important.
# exist_ok=True means Python will not complain
# if the folder already exists.

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }


# ============================================================
# TRAIN DEMAND PREDICTOR
# ============================================================

def train_demand_model():

    print()
    print("Generating demand dataset...")

    # Generate simulated driving data

    driving_data = generate_driving_cycle(
        n_steps=12000,
        seed=42
    )

    # Create lag features

    dataset = create_demand_dataset(
        driving_data
    )

    # Make absolutely sure the directory exists
    # before saving the CSV.

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save dataset

    dataset.to_csv(

        PROCESSED_DIR /
        "demand_dataset.csv",

        index=False
    )

    print(
        f"Demand dataset created: {len(dataset)} rows"
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    features = [

        "power_lag_1",

        "power_lag_2",

        "power_lag_3",

        "power_lag_4",

        "power_lag_5",

        "power_lag_6",

        "speed_kmh",

        "acceleration_kmh_s"
    ]

    # Target variable

    target = "power_demand_kw"

    X = dataset[features]

    y = dataset[target]

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    split_index = int(
        len(dataset) * 0.80
    )

    X_train = X.iloc[
        :split_index
    ]

    X_test = X.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_test = y.iloc[
        split_index:
    ]

    print()
    print("Training demand predictor...")

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    model = RandomForestRegressor(

        n_estimators=150,

        max_depth=18,

        min_samples_leaf=2,

        random_state=42,

        n_jobs=-1
    )

    # Train

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(

        {
            "model": model,

            "features": features

        },

        MODELS_DIR /
        "demand_predictor.pkl"
    )

    print()
    print("Demand predictor saved.")

    print()
    print("Demand Predictor Metrics")

    print(
        f"MAE  : {metrics['MAE']:.4f}"
    )

    print(
        f"RMSE : {metrics['RMSE']:.4f}"
    )

    print(
        f"R2   : {metrics['R2']:.4f}"
    )

    return metrics


# ============================================================
# TRAIN ENERGY MANAGEMENT POLICY
# ============================================================

def train_policy_model():

    print()
    print("Generating policy dataset...")

    # Generate teacher-labelled dataset

    dataset = create_policy_dataset(

        n_samples=20000,

        seed=42
    )

    # Make sure folder exists

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save dataset

    dataset.to_csv(

        PROCESSED_DIR /
        "policy_dataset.csv",

        index=False
    )

    print(
        f"Policy dataset created: {len(dataset)} rows"
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    features = [

        "demand_kw",

        "battery_soc",

        "supercap_soc"
    ]

    # Target

    target = "battery_power_kw"

    X = dataset[features]

    y = dataset[target]

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    split_index = int(
        len(dataset) * 0.80
    )

    X_train = X.iloc[
        :split_index
    ]

    X_test = X.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_test = y.iloc[
        split_index:
    ]

    print()
    print(
        "Training energy-management policy..."
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    model = RandomForestRegressor(

        n_estimators=200,

        max_depth=14,

        min_samples_leaf=2,

        random_state=42,

        n_jobs=-1
    )

    # Train model

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(

        {
            "model": model,

            "features": features

        },

        MODELS_DIR /
        "student_policy.pkl"
    )

    print()
    print(
        "Student policy saved."
    )

    print()
    print(
        "Student Policy Metrics"
    )

    print(
        f"MAE  : {metrics['MAE']:.4f}"
    )

    print(
        f"RMSE : {metrics['RMSE']:.4f}"
    )

    print(
        f"R2   : {metrics['R2']:.4f}"
    )

    return metrics