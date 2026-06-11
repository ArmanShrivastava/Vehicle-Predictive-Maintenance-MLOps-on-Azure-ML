import argparse
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--train_data", type=str)
parser.add_argument("--model_output", type=str)
parser.add_argument("--num_leaves", type=int, default=31)
parser.add_argument("--learning_rate", type=float, default=0.05)
parser.add_argument("--n_estimators", type=int, default=200)   # reduced for 2-core node
args = parser.parse_args()

#the above block sets up the command-line argument parser to allow for flexible input of training data, model output path, and LightGBM hyperparameters.
#line 8 tells the script to expect a string argument for the training data directory, line 9 for the model output directory, and lines 10-12 for the LightGBM hyperparameters with default values. 

mlflow.autolog()

df = pd.read_csv(Path(args.train_data) / "train.csv")
X, y = df.drop(columns=["failure_30d"]), df["failure_30d"]
#this block reads the training data from the specified directory, separates the features (X) from the target variable (y), which is "failure_30d". 
# The drop method is used to remove the target column from the feature set.

model = lgb.LGBMClassifier(
    num_leaves=args.num_leaves,
    learning_rate=args.learning_rate,
    n_estimators=args.n_estimators,
    class_weight="balanced",
    n_jobs=2,            # match DS2_v2 cores
    random_state=42,
)

model.fit(X, y)
#this block initializes the LightGBM classifier with the specified hyperparameters and fits the model to the training data. 
# The class_weight="balanced" argument is used to handle any class imbalance in the target variable, and n_jobs=2 allows the model to use 2 CPU cores for
training.

mlflow.lightgbm.save_model(model, args.model_output)
print("Model saved")
#finally, this block saves the trained model to the specified output directory using MLflow's Light