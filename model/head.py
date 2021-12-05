import torch
from torch import nn

def create_head(num_features, number_classes, dropout_prob=0.5, activation_func=nn.ReLU,isEff=False):
    """ ***************************************************************************************
    *    Title: Multi-Label
    *    Author: aman5319
    *    Date: 8th March 2019
    *    Availability: https://github.com/aman5319/Multi-Label/blob/master/Classify_scenes.ipynb
    *
    *************************************************************************************** """
    # adding 3 layers
    features_lst = [num_features, num_features//2, num_features//4]
    layers = []
    for in_f, out_f in zip(features_lst[:-1], features_lst[1:]):
        layers.append(nn.Linear(in_f, out_f))
        layers.append(activation_func())
        if isEff==False:
            layers.append(nn.BatchNorm1d(out_f))
        if dropout_prob != 0:
            layers.append(nn.Dropout(dropout_prob))
    layers.append(nn.Linear(features_lst[-1], number_classes))
    return nn.Sequential(*layers)