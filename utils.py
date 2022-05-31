import os
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import torch
import numpy as np
from sklearn.metrics import det_curve

audio_paths = glob.glob('audio/*.npy')
processed_prefix = 'processed/'


def get_paths(option, audio_paths):
    prefixes = []
    if option == 'train':
        prefixes.append('NIS')
        prefixes.append('VIT')
    elif option == 'valid':
        prefixes.append('EDI')
    elif option == 'test':
        prefixes.append('CMU')

    files = []  # containing prefix
    for prefix in prefixes:
        for path in audio_paths:
            if prefix in path:
                files.append(path.split('/')[1])  # only get file name

    return files


def get_data(file_names, split_option):
    audio_arr = []
    label_arr = []
    for file_name in file_names:
        with open('audio/' + file_name, 'rb') as f:
            audio_arr.append(np.load(f))
        with open('labels/' + file_name, 'rb') as f:
            label_arr.append(np.load(f))

    flat_audio = [item for sublist in audio_arr for item in sublist]
    flat_label = [item for sublist in label_arr for item in sublist]

    flat_audio = np.asarray(flat_audio)
    flat_label = np.asarray(flat_label)

    processed_path = processed_prefix + split_option
    audio_path = processed_path + '/audio/'
    label_path = processed_path + '/labels/'

    audioPathExist = os.path.exists(audio_path)
    labelPathExist = os.path.exists(label_path)

    if audioPathExist:
        np.save(audio_path + 'audio.npy', flat_audio)
    else:
        os.makedirs(audio_path)
        np.save(audio_path + 'audio.npy', flat_audio)

    if labelPathExist:
        np.save(label_path + 'labels.npy', flat_label)
    else:
        os.makedirs(label_path)
        np.save(label_path + 'labels.npy', flat_label)


def accuracy(model, data_loader, random=False):
    correct = 0
    total = 0
    # zero_freq = 0
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            # zero_freq += np.count_nonzero(labels)

            # calculate outputs by running images through the network
            if not random:
                outputs = model(inputs)
                outputs = torch.round(torch.sigmoid(outputs))
                total += labels.size(0)
                correct += (outputs == labels).sum().item()
            else:
                outputs = torch.ones(len(labels), 1)
                # the class with the highest energy is what we choose as prediction
                total += labels.size(0)
                correct += (outputs == labels).sum().item()
    # print('There are :', zero_freq / len(data_loader.dataset) * 100, '% of 1s')
    print(f'Accuracy : {100 * correct // total} %')


def get_predictions(model, data_loader):
    y_true = np.zeros(len(data_loader.dataset))
    y_score = np.zeros(len(data_loader.dataset))
    c = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data
            outputs = model(inputs)
            y_true[c:c + len(labels)] = labels.squeeze()
            y_score[c:c + len(outputs)] = outputs.squeeze()
            c += len(labels)

    return y_true, y_score


def DET_curve(model, data_loader):
    y_true, y_score = get_predictions(model, data_loader)
    fpr, fnr, _ = det_curve(y_true, y_score)
    return fpr, fnr


def EER_curve(model, data_loader):
    y_true, y_score = get_predictions(model, data_loader)
    fprs, tprs, _ = roc_curve(y_true, y_score)
    eer = fprs[np.nanargmin(np.absolute((1 - tprs) - fprs))]
    return fprs, tprs, eer


def plot_DET_curve(fprs, fnrs):
    fig = plt.figure(1, (7, 4))
    ax = fig.add_subplot(1, 1, 1)

    for i in range(len(fprs)):
        if i == 0:
            label = 'test'
        elif i == 1:
            label = 'valid'
        ax.plot(fprs[i], fnrs[i], label=label)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('False Negative Rate')

    ax.legend()

    plt.title('DET curves on Test and Valid Sets')
    fig.savefig('images/DET_curve.png', bbox_inches='tight')
    #plt.show()
    plt.close()


def plot_EER_curve(fprs, tprs, eer, dataset=None):
    fig = plt.figure(1, (7, 4))
    ax = fig.add_subplot(1, 1, 1)

    ##ax.plot(fprs, fnrs, label='wtf')

    ax.plot([0,1], [1,0])
    ax.plot(fprs, tprs, label='ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    ax.legend()
    plt.title(f"EER:{eer:.2f}, Dataset:{dataset}")
    fig.savefig(str('images/EER_'+dataset+'.png'), bbox_inches='tight')
    #plt.show()
    plt.close()


