import argparse
from pathlib import Path
import json
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets,transforms
from torchvision import models
from collections import OrderedDict


def prepare_dataloaders(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
    'train': transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'test' : transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid' : transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])}

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=True)}
    return image_datasets, dataloaders

def build_classifier(input_size, output_size, hidden_layers):
    layers = []
    for i in range(len(hidden_layers)):
        if i == 0:
            layers.append(nn.Linear(input_size, hidden_layers[i]))
            layers.append(nn.ReLU())
            in_size, out_size = hidden_layers[i], hidden_layers[i+1]
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
        elif i == len(hidden_layers)-1:
            in_size, out_size = hidden_layers[i], output_size
            layers.append(nn.Linear(in_size, out_size))
        else:
            in_size, out_size = hidden_layers[i], hidden_layers[i+1]
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
    layers.extend([nn.LogSoftmax(dim=1)])
    print(layers)
    model = nn.Sequential(*layers)
    return model

def train_model_and_save(args, dataLoaders):
    # check if gpu available
    cuda = torch.cuda.is_available
    if args.gpu and cuda:
        use_cuda = True
    else:
        use_cuda = False
    # only vgg13 is available
    if args.arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        print(args.arch, " is not implemented yet.")
        exit(1)

    for param in model.parameters():
        param.requires_grad = False

    classifier = build_classifier(25088, len(cat_to_name), [4096, 1024, 512, 256])

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    train_model(model, dataloaders['train'], dataloaders['valid'], criterion, optimizer, args.epochs, use_cuda)

    # check accuracy on test datasets
    test_accuracy = validate(model, dataloaders['test'], use_cuda)

    # save checkpoint file
    data_folder = Path(args.save_dir)
    checkpointfile = data_folder / "my_checkpoint_file.pth"
    checkpoint = {
              'opt_state_dict': optimizer.state_dict(),
              'opt_epoch': args.epochs,
              'classifier_input': 25088,
              'classifier_output': len(cat_to_name),
              'classifier_hidden': [4096, 1024, 512, 256],
              'model': models.vgg13(pretrained=True),
              'class_to_id': model.class_to_idx,
              'state_dict': model.state_dict()}

    torch.save(checkpoint, checkpointfile)
    return checkpointfile, test_accuracy

def train_model(model, training_dataloader, valid_dataloader, criterion, optimizer, epochs = 10, use_cuda=False):
    print_every = 40
    steps = 0
    # change to cuda if
    if use_cuda and torch.cuda.is_available():
        model.to('cuda')
    else:
        model.to('cpu')

    for e in range(epochs):
        running_loss = 0
        running_accuracy = 0
        for ii, (inputs, labels) in enumerate(training_dataloader):
            steps += 1

            if use_cuda and torch.cuda.is_available():
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # stats
            running_loss += loss.item()
            probs = torch.exp(outputs).data
            equality = (labels.data == probs.max(1)[1])
            running_accuracy += equality.type_as(torch.FloatTensor()).mean()

            if steps % print_every == 0:
                # validate accuracy
                valid_loss, valid_accuracy = validate(model, valid_dataloader, criterion, use_cuda)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Running Loss: {:.4f}".format(running_loss/print_every),
                      "Running Accuracy: {:.4f}".format(running_accuracy/print_every),
                      "Valiation Loss: {:.4f}".format(valid_loss),
                      "Validation Accuracy: {:.4f}".format(valid_accuracy)
                     )
                running_loss = 0
                running_accuracy=0


# Implement a function for the validation pass
def validate(model, dataloader, criterion, use_cuda=False):
    model.eval()
    if use_cuda and torch.cuda.is_available():
        model.to('cuda')
    else:
        model.to('cpu')
    accuracy = 0
    loss = 0
    for ii, (images, labels) in enumerate(dataloader):

        if use_cuda:
            images, labels = images.to('cuda'), labels.to('cuda')

        output = model.forward(images)
        loss += criterion(output, labels).item()
        probs = torch.exp(output).data
        equality = (labels.data == probs.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()
    model.train()
    # return loss, accuracy
    return loss / len(dataloader), accuracy / len(dataloader)


## Main body
if __name__ == "__main__":
    usage = '''
    Basic usage:
        python train.py data_directory
    Set directory to save checkpoints:
        python train.py data_dir --save_dir save_directory
    Choose architecture:
        python train.py data_dir --arch "vgg13"
    Set hyperparameters:
        python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    Use GPU for training:
        python train.py data_dir --gpu
    '''

    cwd = os.getcwd()

    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument("data_dir", help="The directory containing the data")
    parser.add_argument("--save_dir", help="The directory for saving the checkpoint", default=cwd)
    parser.add_argument("--arch", help="The arch for the neuronetwork", default="vgg13")
    parser.add_argument("--learning_rate", help="The directory containing the data", type=float, default=0.001)
    parser.add_argument("--hidden_units", help="The directory containing the data", type=int)
    parser.add_argument("--epochs", help="number of epochs", type=int, default=10)
    parser.add_argument("--gpu", type=bool, default=False, help='Use GPU or Not')

    args = parser.parse_args()

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    image_datasets, dataloaders = prepare_dataloaders(args.data_dir)

    checkpointfile, test_accuracy = train_model_and_save(args, dataloaders)

    print("Model test accuracy: ", test_accuracy)
    print("Trained model is saved in ", checkpointfile)
