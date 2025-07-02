# Execute only once!
import os
import sys
import torch
import time
import random
import logging
import json

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.neighbors import NearestNeighbors

from be_great import GReaT
from utils import set_logging_level

# Setup
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Logging & seed
logger = set_logging_level(logging.INFO)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Evaluation Functions
def evaluate_downstream_task(real_data, synthetic_data, target_col, model_type="LR"):
    X_train = synthetic_data.drop(columns=[target_col])
    y_train = synthetic_data[target_col]
    X_test = real_data.drop(columns=[target_col])
    y_test = real_data[target_col]

    if model_type == "LR":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "DT":
        model = DecisionTreeClassifier()
    elif model_type == "RF":
        model = RandomForestClassifier()
    else:
        raise ValueError("Invalid model_type")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if y_test.nunique() > 10:
        return {
            "mse": mean_squared_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred)
        }
    else:
        return {
            "accuracy": accuracy_score(y_test, y_pred)
        }

def evaluate_discriminator(real_data, synthetic_data):
    real_data = real_data.copy()
    synthetic_data = synthetic_data.copy()
    real_data["__label__"] = 0
    synthetic_data["__label__"] = 1
    combined = pd.concat([real_data, synthetic_data])
    X = combined.drop(columns="__label__")
    y = combined["__label__"]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    y_pred = clf.predict(X)

    return {"discriminator_accuracy": accuracy_score(y, y_pred)}

def compute_dcr_score(real_data, synthetic_data):
    scaler = MinMaxScaler()
    X_real = scaler.fit_transform(real_data)
    X_syn = scaler.transform(synthetic_data)
    nn = NearestNeighbors(n_neighbors=1, metric='manhattan')
    nn.fit(X_real)
    distances, _ = nn.kneighbors(X_syn)
    return {
        "avg_dcr_l1": float(distances.mean()),
        "std_dcr_l1": float(distances.std())
    }

# Config
epochs = 50
batch_size = 16
sample_num = 5000

dataset_configs = {
    "california": epochs,
    "insurance": epochs,
    "adult": epochs,
    "heloc": epochs,
    "covtype": epochs,
    "intrusion": epochs,
}

dataset_configs = {
    "california": epochs,
    "insurance": epochs,
    "adult": epochs,
    "heloc": epochs,
    "covtype": epochs,
    "intrusion": epochs,
}

column_dict = {
    "iris": ["sepal length", "sepal width", "petal length", "petal width", "target"],
    "california": [],
    "insurance": [],
    "adult": [],
    "heloc": [],
    "covtype": [],
    "intrusion": []
}

# Loop through datasets
for dataset_name, n_epochs in dataset_configs.items():
    logger.info(f"--- Starting training for dataset: {dataset_name} ---")
    output_dir = f"sample{sample_num}_epochs{n_epochs}_seed{SEED}"
    os.makedirs(f"outputs/{output_dir}", exist_ok=True)

    # Load dataset
    if dataset_name == "california":
        data = datasets.fetch_california_housing(as_frame=True).frame
    elif dataset_name == "insurance":
        data = pd.read_csv("dataset/Insurance_compressed.csv")
    elif dataset_name == "adult":
        data = datasets.fetch_openml("adult", version=2, as_frame=True).frame
    elif dataset_name == "heloc":
        data = pd.read_csv("dataset/heloc_dataset_v1.csv")
    elif dataset_name == "covtype":
        data = datasets.fetch_covtype(as_frame=True).frame
    elif dataset_name == "intrusion":
        data = datasets.fetch_kddcup99(as_frame=True, percent10=True).frame
    else:
        logger.warning(f"Dataset {dataset_name} not found.")
        continue

    if column_dict[dataset_name]:
        data.columns = column_dict[dataset_name]

    # Identify target column
    target_col = (
        "target" if "target" in data.columns else
        "class" if "class" in data.columns else
        "RiskPerformance" if "RiskPerformance" in data.columns else
        "Cover_Type" if "Cover_Type" in data.columns else
        None
    )
    if not target_col:
        logger.warning(f"Unknown target column for dataset: {dataset_name}")
        continue

    # Stratify
    stratify_col = data[target_col] if target_col in data.columns else None

    # Split
    train_data, temp_data = train_test_split(
        data, test_size=0.2, random_state=SEED, stratify=stratify_col
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=SEED, stratify=temp_data[target_col]
    )
    train_data = train_data.head(sample_num)
    logger.info(f"Train size: {len(train_data)}, Val size: {len(val_data)}, Test size: {len(test_data)}")

    # Train GReaT
    experiment_dir = f"outputs/{output_dir}/trainer_{dataset_name}"
    great = GReaT(
        llm="distilgpt2",
        epochs=n_epochs,
        save_steps=1000,
        logging_steps=500,
        experiment_dir=experiment_dir,
        batch_size=batch_size
    )
    start_time = time.time()

    if dataset_name in ["california", "insurance"]:
        early_stopping_metric = "mse" 
        early_stopping_mode = "min"
    else:
        early_stopping_metric = "accuracy" 
        early_stopping_mode = "max"

    trainer = great.fit(
        train_data=train_data,
        eval_data=val_data,
        early_stopping_patience=3,
        early_stopping_metric=early_stopping_metric,
        early_stopping_mode=early_stopping_mode,
        )
    elapsed = time.time() - start_time

    # Save model
    save_path = f"outputs/{output_dir}/model_{dataset_name}"
    great.save(save_path)

    # Sample synthetic data
    samples = great.sample(1000, k=50, device="cuda:0")
    samples.to_csv(f"outputs/{output_dir}/{dataset_name}_samples.csv")

    # Evaluate
    training_log = {
        "dataset": dataset_name,
        "epochs": n_epochs,
        "batch_size": batch_size,
        "seed": SEED,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "training_time_sec": round(elapsed, 2),
        "model_save_path": save_path,
        "samples_path": f"outputs/{output_dir}/{dataset_name}_samples.csv",
        "start_time": start_time,
        "end_time": time.time()
    }

    if hasattr(trainer, "metrics"):
        training_log["metrics"] = trainer.metrics

    # Evaluation Metrics
    training_log["downstream_performance"] = {}
    for model in ["LR", "DT", "RF"]:
        training_log["downstream_performance"][model] = {
            "val": evaluate_downstream_task(val_data, samples, target_col, model),
            "test": evaluate_downstream_task(test_data, samples, target_col, model)
        }

    training_log["discriminator_eval"] = {
        "val": evaluate_discriminator(val_data.drop(columns=[target_col]), samples.drop(columns=[target_col])),
        "test": evaluate_discriminator(test_data.drop(columns=[target_col]), samples.drop(columns=[target_col]))
    }

    training_log["distribution_alignment"] = {
        "val": compute_dcr_score(val_data.drop(columns=[target_col]), samples.drop(columns=[target_col])),
        "test": compute_dcr_score(test_data.drop(columns=[target_col]), samples.drop(columns=[target_col]))
    }

    # Save logs
    log_file = f"outputs/{output_dir}/training_log.json"
    with open(log_file, "w") as f:
        json.dump(training_log, f, indent=4)

    logger.info(f"✅ Finished training for {dataset_name}")
    logger.info(f"📁 Model saved to: {save_path}")
    logger.info(f"📝 Training log saved to: {log_file}")