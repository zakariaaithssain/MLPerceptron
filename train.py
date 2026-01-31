import torch
import joblib
import logging

from mlp import MultiLayerPerceptron
from config import READY_DATA_PATH

from sklearn.model_selection import train_test_split



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

num_features = X_train.size(dim=1)
model = MultiLayerPerceptron(input_dim=num_features)
dataset = torch.utils.data.TensorDataset(X_train, y_train)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=37, shuffle=True,
                                            pin_memory=True) #in case sehel 3lina Allah bshy GPU
num_epochs = 500
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def train(model, dataloader, num_epochs:int, lr:float, device) -> list[dict]: 
        #move model to device
        model.to(device)
        #Adam with L2 regularization, regularization strength 1e-4
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if param.ndim == 1:  #bias
                no_decay.append(param)
            else: #weights
                decay.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": 1e-4},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr
        )

        criterion = torch.nn.CrossEntropyLoss()
        best_val_loss = float("inf")
        patience = 20
        patience_counter = 0

        logging.info("training started.")
        for epoch in range(num_epochs): 
                model.train()
                train_loss = 0.0
                n_batches = 0
                for X_batch, y_batch in dataloader: 
                        # to calculate the mean of loss per batch (train_loss/n_batches)
                        n_batches+=1
                        #tensors should be at the same device as the model
                        X_batch = X_batch.to(device)
                        y_batch = y_batch.to(device)
                        #forward pass
                        logits = model(X_batch)
                        batch_loss = criterion(logits, y_batch)

                        train_loss+= batch_loss.item()

                        #backward pass
                        optimizer.zero_grad()
                        batch_loss.backward()

                        #update params 
                        optimizer.step()
                train_loss/= n_batches

                # validation for early stopping
                val_loss = validate(model, criterion, device)

                logging.debug(
                    f"epoch {epoch:03d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}"
                )

                # early stopping
                if val_loss <= best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    logging.debug(f"patience counter reinitialized.")
                    #save the state of the best model so far, allows as to increase patience, that will prevent stopping because of noise. 
                    torch.save(
                        {"best_model_state":model.state_dict(),
                        "best_validation_loss": best_val_loss,
                        "epoch":epoch}, 
                                 "best_state.pt")
                else:
                    patience_counter += 1
                    logging.debug(f"validation loss did not improve at epoch {epoch}")

                if patience_counter >= patience:
                    logging.warning(f"early stopping triggered after {patience} waiting epochs. best state saved to 'best_state.pt'")
                    break

                if epoch % 10 == 0 or epoch==99: 
                    logging.info(f"epoch {epoch:03d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")



@torch.no_grad()
def validate(model, criterion, device):
    "to be used for early stopping inside `train`"

    model.eval()
    total_loss = 0.0
    num_batches = 0

    dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=37, shuffle=True,
                                            pin_memory=True) #in case we have GPU
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches
                  


if __name__ == "__main__": 

    try:
        train(model, dataloader, num_epochs, lr, device)
        exit(0)
    except KeyboardInterrupt: 
          logging.warning("training interrupted manually.")
          exit(1)
    
    except Exception as e: 
          logging.error(str(e))
          exit(1)