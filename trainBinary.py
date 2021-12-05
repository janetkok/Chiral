from torch.optim import lr_scheduler
import torch.optim as optim
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader

from data import BinaryDataset
from parser import parse_args_train_binary
from utils import seed_worker

from timm.utils import *
from timm.models import create_model, resume_checkpoint, load_checkpoint

from sklearn.model_selection import StratifiedKFold
from tqdm import trange

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


def train(model, data_loader, criterion, optimizer, scheduler, num_epochs=5,saver=None,foldMain=''):
  
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

          acc1,_ = accuracy(output, target, topk=(1,2))


          if phase == "train":
            # zero the grad to stop it from accumulating
            optimizer.zero_grad()
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # update the model parameters
            optimizer.step()
            

        running_loss += loss.item() * data.size(0)
        running_corrects += acc1.item() * data.size(0)

      epoch_loss = running_loss / len(data_loader[phase].dataset)
      epoch_acc = running_corrects / len(data_loader[phase].dataset)
       
      # monitor learning rate
      lrl = [param_group['lr'] for param_group in optimizer.param_groups]
      lr = sum(lrl) / len(lrl)

      result.append('{} LR: {:.4f} Loss: {:.4f} Acc: {:.4f} '.format(
        phase, lr,epoch_loss, epoch_acc))
      
      if phase=="train":
          train_metrics = OrderedDict([('loss', epoch_loss), ('acc', epoch_acc)])
      else:
          eval_metrics = OrderedDict([('loss', epoch_loss), ('acc', epoch_acc)])
          if saver is not None:
              # save proper checkpoint with eval metric
              save_metric = epoch_acc
              best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)  
          if best_metric is not None:
              print('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

    update_summary(epoch, train_metrics, eval_metrics, filename=os.path.join(
      foldMain, 'summary.csv'), write_header=(epoch==0))

    print(result)
  return eval_metrics


def main():
  seed = 0
  os.environ['PYTHONHASHSEED']=str(seed)
  seed_worker(seed)
  g = torch.Generator()
  g.manual_seed(seed)

  args, args_text = parse_args_train_binary()
  decreasing = True if args.eval_metric == 'loss' else False
  foldMain = ''
  fold=-1
  cv_metrics = dict(loss=[],acc=[])
   
  transform = torchvision.transforms.Compose([torchvision.transforms.Resize((320, 320)),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(args.mean, args.std)
                                ])

  ds = pd.read_csv(args.dataset)
  label = np.array(ds.drop(['image_path'],axis=1))
  splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

  for train_idx, test_idx in splitter.split(ds['image_path'], label):
      fold = fold+1

      print('fold '+ str(fold))
      exp_name = '-'.join([
                    str(fold),
                    datetime.now().strftime("%Y%m%d-%H%M%S")
                ])
      output_dir = get_outdir(args.output if args.output else './outputBinary/'+args.folder_name, exp_name)
      print(output_dir)

      if fold == 0:
        foldMain = output_dir

      trainset = BinaryDataset(ds, train_idx, transparent2white = args.transparent2white, 
                    color2grayscale = args.color2grayscale, transforms=transform)
      valset = BinaryDataset(ds, test_idx,  transparent2white = args.transparent2white,  
                        color2grayscale = args.color2grayscale, transforms=transform)

      print(f"trainset len {len(trainset)} valset len {len(valset)}")

      dataloader = {"train": DataLoader(trainset, shuffle=True,
       batch_size=args.batch_size,drop_last=True,num_workers=8, worker_init_fn=seed_worker, generator=g),
                  "val": DataLoader(valset, shuffle=False, batch_size= 1,drop_last=False,num_workers=8, worker_init_fn=seed_worker, generator=g),
                    }
      print(f"train loader len {len(dataloader['train'].sampler)} valset len {len(dataloader['val'].sampler)}")

      model = create_model(
          args.model,
          num_classes=args.pretrain_num_classes,
          checkpoint_path=args.initial_checkpoint
          )

      # fine-tne top layers
      if args.freeze:
        ct = 0
        freezeN = 8 if args.model=="tv_resnet50" else 5
        for child in model.children():
          ct += 1
          if ct < freezeN:
              for param in child.parameters():
                  param.requires_grad = False

      if args.model=="tv_resnet50":
          num_ftrs = model.fc.in_features
          model.fc = nn.Linear(num_ftrs, args.num_classes,bias=True)
      elif args.model=="efficientnetv2_m":
          num_ftrs = model.classifier.in_features
          model.classifier = nn.Linear(num_ftrs, args.num_classes,bias=True)

      model = model.cuda(0)
      
      # loss
      criterion = nn.CrossEntropyLoss().cuda()

      # specify optimizer
      optimizer = optim.Adam(model.parameters(), lr=args.lr)
      sgdr_partial = lr_scheduler.CosineAnnealingLR(
          optimizer, T_max=20, eta_min=0.0005)
      
      saver = CheckpointSaver(model=model, optimizer=optimizer, args=args, model_ema=None, amp_scaler=None,
        checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
      
      # save training config
      with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
          f.write(args_text)

      eval_metrics = train(model,dataloader , criterion, optimizer,sgdr_partial,
                      num_epochs=args.epochs,saver=saver,foldMain=foldMain)
      cv_metrics['loss'].append(eval_metrics['loss'])
      cv_metrics['acc'].append(eval_metrics['acc'])

  # cross validation results
  avg_metrics = OrderedDict([('loss', np.mean(cv_metrics['loss'])),('acc', np.mean(cv_metrics['acc']))])
  print(avg_metrics)
  update_cv('avg_metrics',avg_metrics,  os.path.join(foldMain, 'summary.csv'), write_header=True)
  
  std_metrics =  OrderedDict([ ('loss', np.std(cv_metrics['loss'])), ('acc', np.std(cv_metrics['acc']))])
  print(std_metrics)
  update_cv('std_metrics',std_metrics, os.path.join(foldMain, 'summary.csv'), write_header=True)

if __name__ == '__main__':
    main()

