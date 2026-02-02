# Multi Layer Perceptron Lab (ENSIAS S4 2IA's MAJOR)

this project implements, trains, validates, and tests a Multi Layer Perceptron structure using Pytorch, respecting the specifications of **1st Deep Learning Lab of 2IA's major at ENSIAS (4th semester)**.  

### architecture (specified in the lab):   
- input layer : input_dim features (dynamic)  
- 1st fully conneced hidden layer : 32 neurons + ReLU activation  
- 2nd fully connected hidden layer : 16 neurons + ReLU activation  
- output layer : 2 classes (no activation, we use cross entropy as the criterion, which converts logits to probas automatically)  

### training data:  
we use the `Heart Disease UCI dataset` (over 900 data points), after preprocessing it, see `data/preprocess.ipynb`  


### metrics: 
since our task is a binary classification for maximising the heart disease detection (minimizing False Negatives), we focus on the `Recall metric` to evaluate the model in the testing phase.  
