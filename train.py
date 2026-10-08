from pathlib import Path
import json

from src.train_models import (
    train_demand_model,
    train_policy_model
)


# ============================================================
# CREATE PROJECT DIRECTORIES
# ============================================================

directories = [

    Path("data"),

    Path("data/raw"),

    Path("data/processed"),

    Path("models"),

    Path("results"),

    Path("results/figures"),

    Path("results/metrics")
]


for directory in directories:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# START
# ============================================================

print()
print("================================")
print("       HESS ML TRAINING")
print("================================")
print()


# ============================================================
# DEMAND PREDICTOR
# ============================================================

print(
    "STEP 1: Training demand predictor..."
)

demand_metrics = train_demand_model()


print()
print(
    "Demand predictor training completed."
)


# ============================================================
# ENERGY MANAGEMENT POLICY
# ============================================================

print()
print(
    "STEP 2: Training energy-management policy..."
)

policy_metrics = train_policy_model()


print()
print(
    "Energy-management policy training completed."
)


# ============================================================
# COMBINE RESULTS
# ============================================================

results = {

    "demand_predictor":
        demand_metrics,

    "student_policy":
        policy_metrics
}


# ============================================================
# SAVE METRICS
# ============================================================

metrics_path = Path(
    "results/metrics/model_metrics.json"
)


with open(
    metrics_path,
    "w"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("================================")
print(" ALL MODELS TRAINED SUCCESSFULLY")
print("================================")
print()


print(
    json.dumps(
        results,
        indent=4
    )
)


print()
print("Generated files:")
print()
print("✓ data/raw/driving_cycle.csv")
print("✓ data/processed/demand_dataset.csv")
print("✓ data/processed/policy_dataset.csv")
print("✓ models/demand_predictor.pkl")
print("✓ models/student_policy.pkl")
print("✓ results/metrics/model_metrics.json")
print()