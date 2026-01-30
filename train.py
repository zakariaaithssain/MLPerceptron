import torch
import joblib
import logging

from mlp import MultiLayerPerceptron
from config import READY_DATA_PATH





def train(model, dataloader, num_epochs:int, lr:float, device) -> list[dict]: 
        #move model to device
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        logging.info("training started.")
        for epoch in range(num_epochs): 
                model.train()
                epoch_loss = 0.0
                for X_batch, y_batch in dataloader: 
                        #tensors should be at the same device as the model
                        X_batch.to(device)
                        y_batch.to(device)

                        #forward pass
                        logits = model(X_batch)
                        batch_loss = criterion(logits, y_batch)

                        epoch_loss+= batch_loss.item()

                        #backward pass
                        optimizer.zero_grad()
                        batch_loss.backward()

                        #update params 
                        optimizer.step()

                logger.debug(f"epoch: {epoch}; cross entropy loss: {epoch_loss:.4f}")

                if epoch % 10 == 0: 
                    logger.info(f"epoch: {epoch}; cross entropy loss: {epoch_loss:.4f}")

        
                        


if __name__ == "__main__": 
    logger = logging.Logger("train")
    data = joblib.load(READY_DATA_PATH)

    training_data = data["train"]
    X = training_data["X"]
    y = training_data["y"]
    #convert to tensors
    X = torch.tensor(X, dtype=torch.float32) #float is required for gradient calcs
    y = torch.tensor(y, dtype=torch.long) #long type is issential for cross entropy

    num_features = X.size(dim=1)
    model = MultiLayerPerceptron(input_dim=num_features)
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=37, shuffle=True,
                                              pin_memory=True) #in case sehel 3lina Allah bshy GPU
    num_epochs = 100
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        train(model, dataloader, num_epochs, lr, device)
        exit(0)
    except KeyboardInterrupt: 
          logger.warning("training interrupted manually.")
          exit(1)
    
    except Exception as e: 
          logging.error(str(e))
          exit(1)