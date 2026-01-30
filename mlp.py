from torch import nn 


class MultiLayerPerceptron(nn.Module): 
    
    def __init__(self, input_dim):
        super().__init__()

        self.fc1 = nn.Linear(in_features=input_dim, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=2)

        self.activation = nn.ReLU()

    


    def forward(self, x): 
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x)) 
        logits = self.fc3(x)
        return logits
