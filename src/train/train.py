import torch
import logging

from mlp import MultiLayerPerceptron
from functions import * 
from config import X_train, X_valid, y_train, y_valid





if __name__ == "__main__": 

    num_features = X_train.size(dim=1)
    model = MultiLayerPerceptron(input_dim=num_features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=37, shuffle=True,
                                                pin_memory=True) #in case sehel 3lina Allah bshy GPU
    valid_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=37, shuffle=True,
                                            pin_memory=True) 
    num_epochs = 500
    lr = 1e-3

    try:
        train(model, train_dataloader, num_epochs, lr, device, valid_dataloader)
        exit(0)
    except KeyboardInterrupt: 
          logging.warning("training interrupted manually.")
          exit(1)
    
    except Exception as e: 
          logging.error(str(e))
          exit(1)