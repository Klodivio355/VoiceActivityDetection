import numpy as np
import torch
from dataset import VoiceActivityDetection
from torch.utils.data import DataLoader
from models import MLP
import torch.optim as optim
import torch.nn as nn
from EarlyStopping import EarlyStopping
import matplotlib.pyplot as plt
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def train(hidden_dim_1, hidden_dim_2, learning_rate=0.0001, epochs=15, load_weights=False):

    train_dataset = VoiceActivityDetection(option='train')
    valid_dataset = VoiceActivityDetection(option='valid')
    test_dataset = VoiceActivityDetection(option='test')

    batch_size = 32

    model = MLP(input_dim=13, hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2, output_dim=1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    patience = 2

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # initialize the early_stopping object
    path_name = './mlp_' + str(hidden_dim_1) + '_' + str(hidden_dim_2) + '.pth'
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path_name)

    if not load_weights:
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            model.train()
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                running_loss += loss.item()
                if i % 4000 == 3999:  # print every 4000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 4000:.3f}')
                    running_loss = 0

            model.eval()
            for y, valid_data in enumerate(valid_loader, 0):
                inputs, labels = valid_data
                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_losses.append(loss.item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            model.train()
            scheduler.step(valid_loss)
            model.eval()
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            print('Epoch :', epoch, ' Train Loss :', train_loss, 'Valid Loss :', valid_loss)

            # clear lists to track next epoch
            train_losses = []
            valid_losses = []

            # early_stopping needs the validation loss to check if it has decreased,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        print('Finished Training')

        # This Plotting method is directly imported from the code below :
        # https://github.com/Bjarten/early-stopping-pytorch/blob/master

        # visualize the loss as the network trained
        fig = plt.figure(figsize=(10, 8))
        plt.plot(range(1, len(avg_train_losses) + 1), avg_train_losses, label='Training Loss')
        plt.plot(range(1, len(avg_valid_losses) + 1), avg_valid_losses, label='Validation Loss')

        # find position of lowest validation loss
        minposs = avg_valid_losses.index(min(avg_valid_losses)) + 1
        plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.ylim(0, 0.5)  # consistent scale
        plt.xlim(0, len(avg_train_losses) + 1)  # consistent scale
        plt.grid(True)
        plt.title('Loss over Training on Train/Valid Sets')
        plt.legend()
        plt.tight_layout()
        #plt.show()
        fig.savefig('images/loss_plot.png', bbox_inches='tight')


    path_exists = os.path.exists(path_name)
    print('Weights Path found? ', path_exists)
    if not path_exists:
        print('Weights Path not found, we therefore terminate')
        sys.exit(0)
    model.load_state_dict(torch.load(path_name))

    loaders_list = [train_loader, valid_loader, test_loader]

    for i, loader in enumerate(loaders_list):
        if i == 0:
            dataset_name = 'TRAIN'
        elif i == 1:
            dataset_name = 'VALIDATION'
        else:
            dataset_name = 'TEST'
        model.eval()
        losses = []
        for y, data in enumerate(loader, 0):
            inputs, labels = data
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
        print('Final Loss on :', dataset_name, ' dataset',  np.average(losses))


if __name__ == '__main__':
    if len(sys.argv) == 4:
        hidden_dim_1 = sys.argv[1]
        hidden_dim_2 = sys.argv[2]
        load_weights = sys.argv[3]
        if load_weights == 'True':
            dont_train = True
        else:
            print('Something went wrong')
        train(hidden_dim_1=int(hidden_dim_1), hidden_dim_2=int(hidden_dim_2), load_weights=dont_train)
    elif len(sys.argv) == 3: # TESTED
        hidden_dim_1 = sys.argv[1]
        hidden_dim_2 = sys.argv[2]
        train(hidden_dim_1=int(hidden_dim_1), hidden_dim_2=int(hidden_dim_2))
    elif len(sys.argv) == 5:
        hidden_dim_1 = sys.argv[1]
        hidden_dim_2 = sys.argv[2]
        learning_rate = sys.argv[3]
        epochs = sys.argv[4]
        train(hidden_dim_1=int(hidden_dim_1), hidden_dim_2=int(hidden_dim_2), learning_rate=int(float(learning_rate)),
              epochs=int(epochs))
    else:
        print('Something went wrong')
        sys.exit(0)




