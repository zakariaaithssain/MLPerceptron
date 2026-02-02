import logging
import torch 
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split

READY_DATA_PATH = Path("data/ready/ready.joblib")
TRAINED_MODEL_PATH = Path("train/best_state.pt")
LOG_LEVEL = logging.DEBUG




try:
    data = joblib.load(READY_DATA_PATH)
except FileNotFoundError: 
        print(f"{READY_DATA_PATH} not found.")
        exit(1)

training_data = data["train"]
X_train = training_data["X"]
y_train = training_data["y"]

#we split test data to valid and test, valid wil be used for early stopping. 
X_test, X_valid, y_test, y_valid = train_test_split(data["test"]["X"], data["test"]["y"], test_size=0.5, random_state=123)

#convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32) #float is required for gradient calcs
y_train = torch.tensor(y_train, dtype=torch.long) #long type is issential for cross entropy

X_valid = torch.tensor(X_valid, dtype=torch.float32)
y_valid = torch.tensor(y_valid, dtype=torch.long)


#logging config 
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', 
    handlers=[
        logging.FileHandler("train.log", mode="w"), 
        logging.StreamHandler()
    ], 
    level=LOG_LEVEL
)