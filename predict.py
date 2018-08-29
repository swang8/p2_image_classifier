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

def load_checkpoint(filepath, args):
    if torch.cuda.is_available() and args.gpu:
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location='cpu')

    model = None
    if checkpoint['model_name'] == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif checkpoint['model_name'] == 'alexnet':
        model = models.alexnet(pretrained=True)
    model.class_to_idx = checkpoint['class_to_id']
    model.classifier = build_classifier(checkpoint['classifier_input'], checkpoint['classifier_output'], checkpoint['classifier_hidden'])
    model.load_state_dict(checkpoint['state_dict'])
    return model

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
    #print(layers)
    model = nn.Sequential(*layers)
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    from PIL import Image
    im = Image.open(image)
    orignal_size = im.size

    # re-size, keep the aspect ratio
    new_size = 256
    if orignal_size[0] < orignal_size[1]:
        size = (new_size, orignal_size[1] * new_size / orignal_size[0])
    else:
        size = (orignal_size[0] * new_size / orignal_size[1], new_size)
    im.thumbnail(size)
    #print(im.size)

    # crop center of 224
    weight = 224
    height = 224
    left = (size[0] - weight)/2
    top = (size[1] - height)/2
    right = (size[0] + weight)/2
    bottom = (size[1] + height)/2
    im = im.crop((left, top, right, bottom))
    #plt.imshow(im)

    # to np
    np_img = np.array(im)
    # convert values
    np_img = np_img/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_img = (np_img - mean)/std

    np_img = np_img.transpose(2,0,1)
    #print(np_img.shape)
    return np_img

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    model.eval()
    model.to('cpu')
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    #print(idx_to_class)
    np_img = torch.FloatTensor([process_image(image_path)])
    output = model.forward(np_img)
    probs = torch.exp(output).data.numpy()[0]

    topk_index = np.argsort(probs)[-topk:][::-1]
    topk_class = [idx_to_class[x] for x in topk_index]
    topk_probs = probs[topk_index]

    return topk_probs, topk_class


if __name__ == "__main__":
    usage = '''
    Basic usage:
    python predict.py /path/to/image checkpoint

    Options:
    Return top K most likely classes:
    python predict.py input checkpoint --top_k 3

    Use a mapping of categories to real names:
    python predict.py input checkpoint --category_names cat_to_name.json

    Use GPU for inference:
    python predict.py input checkpoint --gpu
    '''
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument("img_path", help="The full path to the image")
    parser.add_argument("checkpoint", help="The full path to the checkpoint file")
    parser.add_argument("--category_names", help="A json file used for mapping of categories to real names")
    parser.add_argument("--top_K", type=int, help='show top K most likely classes', default=1)
    parser.add_argument("--gpu", type=bool, default=False, help='Use GPU or Not')

    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint, args)

    topk_probs, topk_class = predict(args.img_path, model, args.top_K)
    print()
    print("Predicted class: ", topk_class)
    if args.category_names is not None:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        topk_class_names = [cat_to_name[i] for i in topk_class]
        print("Predicted class names: ", topk_class_names)
    print("Probability: ", topk_probs)
