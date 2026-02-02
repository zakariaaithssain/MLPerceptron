import logging
import logging.config
import torch 
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split


READY_DATA_PATH = Path("data/ready/ready.joblib")
DOTPT_FILE = Path("best_state.pt")

TRAIN_LOGGING_LVL = "INFO"
TEST_LOGGING_LVL = "INFO"


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
y_train = torch.tensor(y_train, dtype=torch.long) #long type is issential for cross entropy criterion

X_valid = torch.tensor(X_valid, dtype=torch.float32)
y_valid = torch.tensor(y_valid, dtype=torch.long)

X_test = torch.tensor(X_test, dtype= torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

#logging config 

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,

    "formatters": {
        "standard": {
            "format": "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
        }
    },

    "handlers": {
        "train_file": {
            "class": "logging.FileHandler",
            "filename": "train.log",
            "mode": "w",
            "formatter": "standard",
            "level": TRAIN_LOGGING_LVL,
        },
        "test_file": {
            "class": "logging.FileHandler",
            "filename": "test.log",
            "mode": "w",
            "formatter": "standard",
            "level": TEST_LOGGING_LVL,
        }
    },

    "loggers": {
        "train": {
            "handlers": ["train_file"],
            "level": TRAIN_LOGGING_LVL,
            "propagate": False,
        },
        "test": {
            "handlers": ["test_file"],
            "level": TEST_LOGGING_LVL,
            "propagate": False,
        }
    }
}

def setup_logging():
    "set up logging config"
    logging.config.dictConfig(LOGGING_CONFIG)
