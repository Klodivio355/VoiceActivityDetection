import torch
from models import MLP
from dataset import VoiceActivityDetection
from torch.utils.data import DataLoader
from utils import plot_DET_curve, plot_EER_curve, accuracy, DET_curve, EER_curve
import sys
import os


def evaluate(hidden_dim_1, hidden_dim_2):
    model = MLP(input_dim=13, hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2, output_dim=1)

    path_name = './mlp_' + str(hidden_dim_1) + '_' + str(hidden_dim_2) + '.pth'

    path_exists = os.path.exists(path_name)
    print('Weights Path found? ', path_exists)
    if not path_exists:
        print('Weights Path not found, we therefore terminate')
        sys.exit(0)

    model.load_state_dict(torch.load(path_name))

    test_dataset = VoiceActivityDetection(option='test')
    valid_dataset = VoiceActivityDetection(option='valid')

    batch_size = 32
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    data_loader_list = [test_loader, valid_loader]
    print('Calculating Accuracy on Test Set')
    accuracy(model, test_loader)
    print('Calculating Accuracy on Valid Set')
    accuracy(model, valid_loader) #Accuracy Valid Set

    print('DET Plot')
    # DET (1 plot with the 2 datasets)
    fprs = []
    fnrs = []
    for data_loader in data_loader_list:
        t_fpr, t_fnr = DET_curve(model, data_loader)
        fprs.append(t_fpr)
        fnrs.append(t_fnr)

    plot_DET_curve(fprs, fnrs)
    print('DET Plot saved under images/DET_curve.png')

    print('EER on Test Dataset')
    # EER (1 plot for each dataset)
    fpr, tpr, eer = EER_curve(model, data_loader_list[0])
    plot_EER_curve(fpr, tpr, eer, dataset="Test")
    print('EER plot saved under images/EER_Test.png')

    print('EER on Valid Dataset')
    fpr, tpr, eer = EER_curve(model, data_loader_list[1])
    plot_EER_curve(fpr, tpr, eer, dataset="Valid")
    print('EER plot saved under images/EER_Valid.png')


if __name__ == '__main__':
    if len(sys.argv) == 3:
        hidden_dim_1 = sys.argv[1]
        hidden_dim_2 = sys.argv[2]
        print('Starting Evaluation')
        evaluate(hidden_dim_1=int(hidden_dim_1), hidden_dim_2=int(hidden_dim_2))
    else:
        print('Something went wrong')
        sys.exit(0)

