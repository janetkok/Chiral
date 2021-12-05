from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from data import FormulatedImbalancedDatasetSampler, MolDataset
from model import create_head
from parser import parse_args_train_multi
from utils import seed_worker

from timm.utils import *
from timm.models import create_model, resume_checkpoint, load_checkpoint

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tqdm import trange
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

import random
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import os
import csv
from collections import OrderedDict
import time
from datetime import datetime



def train(model, data_loader, criterion, optimizer, scheduler, num_epochs=5, saver=None, foldMain=''):

    train_metrics = dict()
    eval_metrics = dict()
    best_metric = None
    best_epoch = None

    for epoch in trange(num_epochs, desc="Epochs"):
        result = []
        for phase in ['train', 'val']:
            if phase == "train":     # training mode
                model.train()
                scheduler.step()
            else:     #  validation mode
                model.eval()

            # keep track of training and validation loss
            running_loss = 0.0
            running_corrects = 0.0

            for data, target in data_loader[phase]:
                # load the data and target to respective device
                data, target = data.cuda(), target.cuda()

                with torch.set_grad_enabled(phase == "train"):
                    # feed the input
                    output = model(data)
                    # calculate the loss
                    loss = criterion(output, target)

                    # prediction for label is true if probability more than 50%
                    preds = torch.sigmoid(output).data > 0.5
                    preds = preds.to(torch.float32)

                    if phase == "train":
                         # zero the grad to stop it from accumulating
                        optimizer.zero_grad()
                        # backward pass: compute gradient of the loss with respect to model parameters
                        loss.backward()
                        # update the model parameters
                        optimizer.step()

                targets_np = target.cpu().detach().to(torch.int).numpy()
                preds_np = preds.cpu().detach().to(torch.int).numpy()

                running_loss += loss.item() * data.size(0)
                running_corrects += accuracy_score(targets_np, preds_np) * data.size(0)

            epoch_loss = running_loss / len(data_loader[phase].sampler)
            epoch_acc = running_corrects / len(data_loader[phase].sampler)

            # monitor learning rate
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            result.append('{} LR: {:.4f} Loss: {:.4f} Acc: {:.4f} '.format(
                phase, lr, epoch_loss, epoch_acc))

            if phase == "train":
                train_metrics = OrderedDict([('loss', epoch_loss), ('acc', epoch_acc)])
            else:
                eval_metrics = OrderedDict([('loss', epoch_loss), ('acc', epoch_acc)])
                if saver is not None:
                    # save proper checkpoint with eval metric
                    save_metric = epoch_acc
                    best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)
                if best_metric is not None:
                    print(
                        '*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

        update_summary(epoch, train_metrics, eval_metrics,filename = os.path.join(
            foldMain, 'summary.csv'), write_header=(epoch == 0))

        print(result)
    return eval_metrics


def main():
    seed = 0
    os.environ['PYTHONHASHSEED'] = str(seed)
    seed_worker(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    args, args_text = parse_args_train_multi()
    decreasing = True if args.eval_metric == 'loss' else False
    foldMain = ''
    fold = -1
    classLabels = ["none", "centre", "axis", "plane"]
    cv_metrics = dict(loss=[], acc=[])

    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((320, 320)),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                    args.mean, args.std)
                                                ])
    transformAug = torchvision.transforms.Compose([torchvision.transforms.Resize((320, 320)),
                                                    torchvision.transforms.RandomAffine(
                                                        0, translate=(0.09, 0.09)),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(
                                                        args.mean, args.std)
                                                    ])

    ds = pd.read_csv(args.dataset)
    label = np.array(ds.drop(['image_path'], axis=1))
    splitter = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    for train_idx, test_idx in splitter.split(ds['image_path'], label):
        fold = fold+1

        print('fold ' + str(fold))
        exp_name = '-'.join([str(fold),
                             datetime.now().strftime("%Y%m%d-%H%M%S")])
        output_dir = get_outdir(args.output if args.output else './outputMulti/'+args.folder_name, exp_name)
        print(output_dir)
        
        if fold == 0:
            foldMain = output_dir

        trainset = MolDataset(ds, train_idx, aug=args.aug, transparent2white = args.transparent2white, 
                    color2grayscale = args.color2grayscale, transforms=transform, transformsAug=transformAug)
        valset = MolDataset(ds, test_idx,  transparent2white = args.transparent2white,  
                        color2grayscale = args.color2grayscale, aug=False, transforms=transform)

        print(f"trainset len {len(trainset)} valset len {len(valset)}")

        dataloader = {"train": DataLoader(trainset, sampler=FormulatedImbalancedDatasetSampler(trainset, minPerct=args.min_perct, addPerct=args.add_perct, maxI=args.maxI, generator=g),
                                          batch_size=args.batch_size, drop_last=True, num_workers=8, worker_init_fn=seed_worker, generator=g),
                      "val": DataLoader(valset, batch_size=1, drop_last=False, num_workers=8, worker_init_fn=seed_worker, generator=g)}

        print(f"train loader len {len(dataloader['train'].sampler)} valset len {len(dataloader['val'].sampler)}")

        model = create_model(
            args.model,
            num_classes=args.pretrain_num_classes,
            checkpoint_path=args.initial_checkpoint
        )
        
        # fine-tne top layers
        if args.freeze:
            if args.model=="tv_resnet50":
                finetuneLayers = ['layer3','layer4','global_pool','fc']

            elif args.model=="efficientnetv2_m":
                finetuneLayers = ['blocks.6','conv_head','bn2','act2','global_pool','classifier']

            for param in model.parameters():
                param.requires_grad = False
            for name, module in model.named_modules():
                if  name in finetuneLayers:
                    for param in module.parameters():
                        param.requires_grad = True
        
        # get the no of on_features in last Linear unit
        num_features = 0
        if args.model == "tv_resnet50":
            num_features = model.fc.in_features
        else:
            num_features = model.classifier.in_features

        # replace the fully connected layer
        top_head = create_head(num_features, len(classLabels),isEff=args.model=="efficientnetv2_m")
        if args.model == "tv_resnet50":
            model.fc = top_head
        else:
            model.classifier = top_head

        model = model.cuda()
        
        # loss
        criterion = nn.BCEWithLogitsLoss().cuda()

        # specify optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        sgdr_partial = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=5, eta_min=0.005)

        saver = CheckpointSaver(model=model, optimizer=optimizer, args=args, model_ema=None, amp_scaler=None,
                                checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
        # save training config
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

        eval_metrics = train(model, dataloader, criterion, optimizer, sgdr_partial,
                             num_epochs=args.epochs, saver=saver, foldMain=foldMain)
        cv_metrics['loss'].append(eval_metrics['loss'])
        cv_metrics['acc'].append(eval_metrics['acc'])
    
    # cross validation results
    avg_metrics = OrderedDict([('loss', np.mean(cv_metrics['loss'])), ('acc', np.mean(cv_metrics['acc']))])
    print(avg_metrics)
    update_cv('avg_metrics', avg_metrics,  os.path.join(foldMain, 'summary.csv'), write_header=True)

    std_metrics = OrderedDict([('loss', np.std(cv_metrics['loss'])), ('acc', np.std(cv_metrics['acc']))])
    print(std_metrics)
    update_cv('std_metrics', std_metrics, os.path.join(foldMain, 'summary.csv'), write_header=True)


if __name__ == '__main__':
    main()
