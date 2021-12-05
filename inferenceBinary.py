from torch.optim import lr_scheduler
import torch.optim as optim
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader

from timm.utils import *
from timm.models import create_model, resume_checkpoint, load_checkpoint

from data import BinaryDataset
from parser import parse_args_infer

import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import os
import csv
from collections import OrderedDict
import time
import yaml
from datetime import datetime

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from sklearn.model_selection import StratifiedKFold
from tqdm import trange


def infer(model, data_loader, output_dir = '',target_module='', gradfile=['']):
    
    phase = 'test'
    outputs = []
    targets = []
    confScores0 = []
    confScores1 = []

    model.eval()
    
    #grad cam setup
    target_layer=None
    for module in model.named_modules():
        if module[0] == target_module:
            target_layer = module[1]
    cam = GradCAMPlusPlus(model=model, target_layer=target_layer, use_cuda=True)
    
    for batch_idx,(data , target) in enumerate(data_loader[phase]):
        # load the data and target to respective device
        data , target = data.cuda() , target.cuda()
        
        with torch.no_grad():
          # feed the input
          output = model(data)

          topk = output.topk(1)[1]
          outputs.append(topk.cpu().numpy())
          targets.append(target.cpu().numpy())
          
          #confidence score
          sm = torch.nn.Softmax(dim = 1)
          outputSM = sm(output)
          confScore0 = round(float(outputSM[:,0]*100),2)
          confScore1 = round(float(outputSM[:,1]*100),2)
          confScores0.append(confScore0)
          confScores1.append(confScore1)

        # gradcam image output
        filename = data_loader[phase].dataset.filename(batch_idx,False)
        filenameBase = data_loader[phase].dataset.filename(batch_idx,True)
        if (filenameBase in gradfile):
          print(filename)

          outfilename = data_loader[phase].dataset.filename(batch_idx,True)
          grayscale_cam = cam(input_tensor=data, target_category=None)
          # Here grayscale_cam has only one image in the batch
          grayscale_cam = grayscale_cam[0, :]
          rgb_img = cv2.imread(filename)
          rgb_img = cv2.resize(rgb_img, (320, 320), interpolation = cv2.INTER_AREA)
          rgb_img = np.float32(rgb_img) / 255

          cam_image = show_cam_on_image(rgb_img, grayscale_cam)
          cv2.imwrite(output_dir+'/'+outfilename, cam_image)

    outputs = np.concatenate(outputs, axis=0)
    targets = np.concatenate(targets, axis=0)

    prec,rec,f1,_ = precision_recall_fscore_support(targets,outputs,average='binary')

    epoch_acc = accuracy_score(targets, outputs)

    with open(os.path.join(output_dir,'./topk_ids.csv'), 'w') as out_file:
        fieldnames = ['img_path', 'pred','gt','correctness','confScore0','confScore1']
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()

        filenames = data_loader[phase].dataset.filenames()
        accCount = 0
        for filename, label,target, confScore0,confScore1  in zip(filenames, outputs,targets,confScores0,confScores1):
            correct = 1
            if int(target)!=int(label[0]):
                correct = 0
            else:
                accCount = accCount + 1 
            writer.writerow({'img_path': filename, 'pred': ','.join([ str(v) for v in label]),'gt': str(target),'correctness': str(correct), 'confScore0': str(confScore0), 'confScore1': str(confScore1)})
    acc = round(accCount/len(targets)*100,4)
    print(acc)
    os.rename(os.path.join(output_dir, './topk_ids.csv'),os.path.join(output_dir, './acc_'+str(acc)+'.csv'))

    eval_metric = OrderedDict([ ('acc', epoch_acc), ('prec', prec), ('recall', rec),('f1', f1)])
    
    return eval_metric


def main():
  seed = 0
  args, args_text = parse_args_infer()
  foldZeroOutputDir = ''
  fold=-1     
  cv_metrics = dict(acc=[],prec=[],recall=[],f1=[])
  num_classes = 2

  transform = torchvision.transforms.Compose([torchvision.transforms.Resize((320, 320)),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(args.mean, args.std)
                                ])

  ds = pd.read_csv(args.dataset)
  label = np.array(ds.drop(['image_path'],axis=1))
  splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
  checkpoint_path_list = sorted(os.listdir(args.checkpoint_dir))
  
  # infer for test set in 10-fold cross val
  for (train_idx, test_idx),checkpoint_path in zip(splitter.split(ds['image_path'], label),checkpoint_path_list):
      fold = fold+1

      print('fold '+ str(fold))
      output_dir = args.checkpoint_dir+'/'+checkpoint_path
      print(output_dir)
      if fold==0:
        foldZeroOutputDir = output_dir

      valset = BinaryDataset(ds, test_idx,  transparent2white = args.transparent2white,  
                  color2grayscale = args.color2grayscale, transforms=transform)

      dataloader = {"test": DataLoader(valset, shuffle=False, batch_size=1)}

      model = create_model(
          args.model
          )

      # replace the fully connected layer
      target_module="layer4.2" #default target module is resnet50
      if args.model=="tv_resnet50":
          num_ftrs = model.fc.in_features
          model.fc = nn.Linear(num_ftrs, num_classes,bias=True)
      else:
          target_module="conv_head" 
          num_ftrs = model.classifier.in_features
          model.classifier = nn.Linear(num_ftrs, num_classes,bias=True)

      load_checkpoint(model, output_dir+"/last.pth.tar")

      model = model.cuda()

      eval_metrics = infer(model,dataloader,output_dir=output_dir,target_module=target_module, gradfile = args.gradfile)
      
      for key,value in eval_metrics.items():
        cv_metrics[key].append(value)

  avg_metrics = OrderedDict([('acc', np.mean(cv_metrics['acc'])), ('prec', np.mean(cv_metrics['prec'])),
  ('recall', np.mean(cv_metrics['recall'])),('f1', np.mean(cv_metrics['f1']))])
  print(avg_metrics)
  update_cv('avg_metrics',avg_metrics,  os.path.join(foldZeroOutputDir, 'summary.csv'), write_header=True)
  
  std_metrics =  OrderedDict([ ('acc', np.std(cv_metrics['acc'])), ('prec', np.std(cv_metrics['prec'])),
                              ('recall', np.std(cv_metrics['recall'])),('f1', np.std(cv_metrics['f1']))])
  print(std_metrics)
  update_cv('std_metrics',std_metrics, os.path.join(foldZeroOutputDir, 'summary.csv'), write_header=True)


if __name__ == '__main__':
    main()

