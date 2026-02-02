from src.functions import test
from src.mlp import MultiLayerPerceptron
from src.config import X_test, y_test, DOTPT_FILE

import torch
import logging


if __name__ == "__main__":  
    input_dim = X_test.size(dim=-1)
    model = MultiLayerPerceptron(input_dim)
    test_data = torch.utils.data.TensorDataset(X_test, y_test)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        test(model, pt_file=DOTPT_FILE, test_data=test_data, device=device)
        exit(0)
    except KeyboardInterrupt: 
          logging.warning("testing interrupted manually.")
          exit(1)
    
    except Exception as e: 
          logging.error(str(e))
          raise
