#COM4511 Speech Technology Assignment (Task 4)

## Voice Activity Detection

This file introduces the structure of the solution and also contains the terminal commands necessary to the replication of the results
presented in the report.

#### REQUIREMENTS
* Have the **audio** and **labels** folder within the root folder.

### 1. Pre-processing Data

The first step is to pre-process our data and split into a Training, Validation and Test set. To do that, run the following line :
**python select_data.py**

This will automatically populate a *processed/* folder within the root folder, with a train/valid/test split with the respective audio samples and labels.

### 2. Training

The training process can be carried out under 3 different command line configurations :
* **python train.py [hidden_layer_1_dimension] [hidden_layer_2_dimension]** (This will be executed with 
the default values found in the train function.
* Example : *python train.py 250 100*
* **python train.py [hidden_layer_1_dimension] [hidden_layer_2_dimension] [learning_rate] [epochs]** (This way, the learning rate and number of
epochs can be changed from the default values)
* Example : *python train.py 250 100 0.001 15*

Alternatively, one may only be interested in the acquisition of the plot of the loss function over training. Therefore, the following
lin can be run :
* **python train.py [hidden_layer_1_dimension] [hidden_layer_2_dimension] [load_weights = True]** 
* Example : *python train.py 250 100 True*

(Ensure a model with the specified dimensions has beeen trained and saved as the weights will be loaded according to those dimensions)

PS : For the implementation of Early Stopping, as well as the acquisition of the loss plot (showing the saved checkpoint of the stopping criterion), pre-existing code has been directly 
imported from the following source : 
*https://github.com/Bjarten/early-stopping-pytorch*

### 3. Evaluation

The evaluationn process can be carried out with the following line:

* **python evaluate.py [hidden_layer_1_dimension] [hidden_layer_2_dimension]**
* Example : *python evaluate.py 250 100*

Again same rule applies, ensure a pre-trained model is available in the root directory as a .pth file. Otherwise, the function will terminate.
