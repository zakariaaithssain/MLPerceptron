from functions import test
from config import X_test, y_test, TRAINED_MODEL_PATH
import torch
import logging


if __name__ == "__main__": 

    trained_model = torch.load(TRAINED_MODEL_PATH)
    test_data = torch.utils.data.TensorDataset(X_test, y_test)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        test(trained_model, test_data, device)
        exit(0)
    except KeyboardInterrupt: 
          logging.warning("testing interrupted manually.")
          exit(1)
    
    except Exception as e: 
          logging.error(str(e))
          exit(1)