import argparse
import pandas as pd
from  pathlib import Path
from sklearn.model_selection import train_test_split

#this parser is used to pass the arguments from the command line when running the script. 
# It allows us to specify the raw data file, the train data directory, and the test data directory. 
# This way, we can easily manage our data files and directories without hardcoding them in the script.
parser = argparse.ArgumentParser()
parser.add_argument("--raw_data", type=str)
parser.add_argument("--train_data", type=str)
parser.add_argument("--test_data", type=str)
args = parser.parse_args()

#this block is responsible for reading the raw data, creating new features, and preparing the dataset for training and testing.
df = pd.read_csv(args.raw_data)
df["temp_pressure_ratio"] = df["engine_temp_c"] / df["oil_pressure_psi"]
df["service_overdue"] = (df["days_since_service"] > 180).astype(int)
df["high_mileage"] = (df["mileage_km"] > 150_000).astype(int)
df = df.drop(columns=["vehicle_id"])

train, test = train_test_split(df, test_size=0.2, stratify=df["failure_30d"], random_state=42)
#the stratifed parameter ensures that the distribution of the target variable (failure_30d) is maintained in both the train and test sets.
Path(args.train_data).mkdir(parents=True, exist_ok=True)
#The mkdir function is used to create the train and test directories if they do not already exist. 
# The parents=True argument allows for the creation of any necessary parent directories, and exist_ok=True prevents an error if the directory already exists.
Path(args.test_data).mkdir(parents=True, exist_ok= True)


train.to_csv(Path(args.train_data) / "train.csv", index= False)
test.to_csv(Path(args.test_data) / "test.csv" , index= False)
#these lines save the train and test datasets as CSV files in the specified directories.
#and index=False is used to prevent pandas from writing row indices to the CSV file, which is often unnecessary for machine learning tasks.

print(f"Train: {len(train)}, Test: {len(test)}")
#Finally, this line prints the number of samples in the train and test datasets, providing a


