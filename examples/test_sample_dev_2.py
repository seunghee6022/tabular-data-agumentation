from be_great import GReaT

import logging
from utils import set_logging_level
from sklearn import datasets

logger = set_logging_level(logging.INFO)
SEED = 42
training_sample_num = 5000
n_samples = 1000

# Configurable dataset list and epochs
dataset_configs = {
    # "iris": 1,
    "adult": 5,
    "california": 5,
    "covtype": 5,
    "heloc": 5,
    "intrusion": 5,
}

for dataset_name, epochs in dataset_configs.items():
    output_dir = f"smaple{training_sample_num}_epochs{epochs}_seed{SEED}"

    great = GReaT.load_from_dir(f"outputs/{output_dir}/model_{dataset_name}")
    # great.load_finetuned_model("../great_private/models/california/california_distilgpt2_100.pt")

    # Continuous column as start
    # data, target = datasets.load_iris(return_X_y=True)
    # sepal = list(data[:, 0])
    # samples = great.sample(20, device="cpu", k=5, start_col="sepal length", start_col_dist=sepal)

    # Random Start
    samples = great.sample(n_samples, device="cpu", k=50)
    # samples = great.sample(n_samples, k=50, device="cuda:1")

    # Categorical column as start
    # samples = great.sample(
    #     20, k=5, start_col="target", start_col_dist={"0.0": 0.33, "1.0": 0.33, "2.0": 0.33}
    # )

    print(samples)
    samples.to_csv(f"outputs/{output_dir}/{dataset_name}_samples.csv")
