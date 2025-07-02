import sys
import os
import time
import random
import logging

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

from be_great import GReaT
from utils import set_logging_level

# Set global logging
logger = set_logging_level(logging.INFO)

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Configurable dataset list and epochs
dataset_configs = {
    # "iris": 1,
    "adult": 5,
    "california": 5,
    "heloc": 5,
    "covtype": 5,
    "intrusion": 5,
}

# Custom column names (only necessary when using numpy or renaming)
column_dict = {
    "iris": ["sepal length", "sepal width", "petal length", "petal width", "target"],
    "adult": [],
    "california": [],
    "heloc": [],
    "covtype": [],
    "intrusion": [],
}

# Loop through datasets
for dataset_name, epochs in dataset_configs.items():
    logger.info(f"--- Starting training for dataset: {dataset_name} ---")
    output_dir = f"train_data_epochs{epochs}_seed{SEED}"

    # Load dataset
    if dataset_name == "iris":
        data = datasets.load_iris(as_frame=True).frame
    elif dataset_name == "adult":
        data = datasets.fetch_openml("adult", version=2, as_frame=True).frame
    elif dataset_name == "california":
        data = datasets.fetch_california_housing(as_frame=True).frame
    elif dataset_name == "covtype":
        data = datasets.fetch_covtype(as_frame=True).frame
    elif dataset_name == "heloc":
        data = pd.read_csv("dataset/heloc_dataset_v1.csv")
    elif dataset_name == "intrusion":
        data = datasets.fetch_kddcup99(as_frame=True, percent10=True).frame
    else:
        logger.warning(f"Dataset {dataset_name} not found.")
        continue

    print(data.head(5))

    # Rename columns if specified
    if column_dict[dataset_name]:
        data.columns = column_dict[dataset_name]

    # Decide stratify column
    if "target" in data.columns:
        stratify_col = data["target"]
    elif dataset_name == "adult":
        stratify_col = data["class"]
    elif dataset_name == "heloc":
        stratify_col = data["RiskPerformance"]
    elif dataset_name == "intrusion":
        stratify_col = None # data["labels"] - ValueError: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.
    elif dataset_name == "covtype":
        stratify_col = data["Cover_Type"]
    else:
        stratify_col = None  # regression or unknown target

    # Split into train/val/test
    train_data, temp_data = train_test_split(
        data, test_size=0.2, random_state=SEED, stratify=stratify_col if stratify_col is not None else None
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=SEED, stratify=stratify_col[temp_data.index] if stratify_col is not None else None
    )

    logger.info(f"Train size: {len(train_data)}, Val size: {len(val_data)}, Test size: {len(test_data)}")
    logger.info(f"data size: {len(data)}")

    # Setup GReaT
    experiment_dir = f"outputs/{output_dir}/trainer_{dataset_name}"
    great = GReaT(
        llm="distilgpt2",
        epochs=epochs,
        save_steps=1000,
        logging_steps=1000,
        experiment_dir=experiment_dir,
    )

    # Start timer
    start_time = time.time()

    # Train
    if isinstance(train_data, pd.DataFrame):
        trainer = great.fit(train_data)
    else:
        column_names = column_dict[dataset_name]
        trainer = great.fit(train_data, column_names=column_names)

    # Stop timer
    elapsed = time.time() - start_time

    # Save model
    save_path = f"outputs/{output_dir}/model_{dataset_name}"
    great.save(save_path)

    # Log training info
    logger.info(f"✅ Finished training for {dataset_name}")
    logger.info(f"🕒 Training time: {elapsed:.2f} seconds")
    logger.info(f"📁 Model saved to: {save_path}")
    logger.info(f"🧪 Epochs: {epochs} | Seed: {SEED}")

