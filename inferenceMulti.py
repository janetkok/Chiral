from torch.optim import lr_scheduler
import torch.optim as optim
import torch
from torch import nn
import torch.nn.functional as F
import torchvision 
from torch.utils.data import Dataset, DataLoader

from timm.utils import *
from timm.models import create_model, resume_checkpoint, load_checkpoint

from data import MolDataset
from parser import parse_args_infer
from model import create_head

import random
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

from tqdm import trange
from sklearn.metrics import accuracy_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def infer(model, data_loader, output_dir = '',target_module='',gradfile=['']):
    
    # initialisation
    phase = 'test'
    predictions = []
    targets = []
    res = [0]*8
    #list of label combinations
    correct_d = {(1,0,0,0):0,(0,1,0,0):0,(0,0,1,0):0,(0,0,0,1):0,(0,1,1,0):0,(0,1,0,1):0,(0,0,1,1):0,(0,1,1,1):0}

    # count for label combination
    df = pd.DataFrame()
    df["label"] = data_loader[phase].dataset.get_labels()
    label_to_count = df["label"].value_counts()
    label_dict = label_to_count.to_dict()
    num_class_d = correct_d.copy()
    for key, value in label_dict.items():
      num_class_d[key] = value

    model.eval()
    
    # grad cam setup
    target_layer=None
    for module in model.named_modules():
        if module[0] == target_module:
            target_layer = module[1]
    cam = GradCAMPlusPlus(model=model, target_layer=target_layer, use_cuda=True)

    for batch_idx,(data , target) in enumerate(data_loader[phase]):
        # load the data and target to respective device
        data , target = data.cuda()  , target.cuda()

        with torch.no_grad():
            # feed the input
            output = model(data)
            
            preds = torch.sigmoid(output).data > 0.5
            pred_np = preds.cpu().numpy().astype(int)
            target_np = target.cpu().numpy().astype(int)

            predictions.append(pred_np)
            targets.append(target_np)
            target_np = target_np[0]
            pred_np = pred_np[0]
            
            if all(target_np == pred_np):
              correct_d[tuple(target_np)]+=1

        # gradcam image output
        filename = data_loader[phase].dataset.filename(batch_idx,False)
        filename_b = data_loader[phase].dataset.filename(batch_idx,True)
        if (filename_b in gradfile):
            print(filename)
            prob = torch.sigmoid(output)[0]*100
            prob = prob.cpu().detach().numpy()
            prob = np.around(np.float32(prob), 2)
            print(prob)
            _, multi_indices = torch.sort(output, descending=True)
            multi_indices = torch.squeeze(multi_indices)
            target_category_multi = [index for index in multi_indices if prob[index] > 50]
            for i, tm in enumerate(target_category_multi):

              outfilename = data_loader[phase].dataset.filename(batch_idx,True)
          
              grayscale_cam = cam(input_tensor=data, target_category=int(tm))
              # Here grayscale_cam has only one image in the batch
              grayscale_cam = grayscale_cam[0, :]
              rgb_img = cv2.imread(filename)
              rgb_img = cv2.resize(rgb_img, (320, 320), interpolation = cv2.INTER_AREA)
              rgb_img = np.float32(rgb_img) / 255
              
              cam_image = show_cam_on_image(rgb_img, grayscale_cam)
              cv2.imwrite(output_dir+'/'+outfilename.split(".png")[0]+'_'+str(i)+'.png', cam_image)

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    epoch_acc = accuracy_score(targets, predictions)
    
    # accuracy per label combination
    num_class = list(num_class_d.values())
    correct = list(correct_d.values())
    for x,(i,j) in enumerate(zip(correct,num_class)):
      if j == 0: # target not present, so assign a number 99 to be filtered out later
        res[x]=99
      else:
        res[x]=i/j

    with open(os.path.join(output_dir,'./predictions.csv'), 'w') as out_file:
        fieldnames = ['img_path', 'pred','gt','correctness']
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()

        filenames = data_loader[phase].dataset.filenames()
        accCount = 0
        for filename, pred,target in zip(filenames, predictions,targets):
            correct = 1
            if not np.all(pred == target):
                correct = 0
            else:
                accCount = accCount + 1 
            writer.writerow({'img_path': filename, 'pred': ','.join([ str(v) for v in pred]),'gt': ','.join([ str(v) for v in target]),'correctness': str(correct)})
    acc = round(accCount/len(targets)*100,4)
    print(acc)
    os.rename(os.path.join(output_dir, './predictions.csv'),os.path.join(output_dir, './acc_'+str(acc)+'.csv'))
    
    # update performance of the fold in detail
    eval_metric = OrderedDict([ ('acc', epoch_acc), ('N_acc',res[0]),('C_acc',res[1]),('A_acc',res[2]),('P_acc',res[3]),('CA_acc',res[4]),('CP_acc',res[5]),('AP_acc',res[6]),('CAP_acc',res[7])])
    update_cv(0,eval_metric, os.path.join(output_dir, 'predictions.csv'), write_header=True)
    
    return eval_metric


def roundDict(numdict):
  "rounding values in dict"
  for key, value in numdict.items():
    if isinstance(value,float):
      numdict[key] = round(value,4)
    if isinstance(value,np.ndarray):
      numdict[keys[i]] = [round(i,4) for i in value]
  return numdict
    
def main():
  seed = 0
  args, args_text = parse_args_infer()
  foldZeroOutputDir = ''
  fold=-1     
  classLabels = ["none", "centre", "axis", "plane"] 
  
  cv_metrics = dict(acc=[],N_acc=[],C_acc=[],A_acc=[],P_acc=[],CA_acc=[],CP_acc=[],AP_acc=[],CAP_acc=[])   

  transform = torchvision.transforms.Compose([torchvision.transforms.Resize((320, 320)),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(args.mean, args.std)
                                ])

  ds = pd.read_csv(args.dataset)
  label = np.array(ds.drop(['image_path'],axis=1))
  splitter = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
  checkpoint_path_list = sorted(os.listdir(args.checkpoint_dir))
  
  # infer for test set in 10-fold cross val
  for (train_idx, test_idx),checkpoint_path in zip(splitter.split(ds['image_path'], label),checkpoint_path_list):
      fold = fold+1

      print('fold '+ str(fold))
      output_dir = args.checkpoint_dir+'/'+checkpoint_path
      print(output_dir)
      if fold==0:
        foldZeroOutputDir = output_dir

      valset = MolDataset(ds, test_idx,transparent2white = args.transparent2white, 
                    color2grayscale = args.color2grayscale, aug=False,transforms=transform)

      dataloader = {"test": DataLoader(valset, shuffle=False, batch_size=1)}

      model = create_model(
          args.model
          )
       
      # compute the no of on_features in last Linear unit
      num_features = 0
      if args.model =="tv_resnet50":
        num_features = model.fc.in_features
      else:
        num_features = model.classifier.in_features

      # replace head
      top_head = create_head(num_features, len(classLabels),isEff=args.model=="efficientnetv2_m") 
      target_module="layer4.2" #default target module is resnet50
      if args.model =="tv_resnet50":
        model.fc = top_head  
      else:
        model.classifier = top_head 
        target_module="conv_head" 
      load_checkpoint(model, output_dir+"/last.pth.tar")

      model = model.cuda()
      
      eval_metrics = infer(model,dataloader,output_dir=output_dir,target_module=target_module,gradfile = args.gradfile)
      
      for key,value in eval_metrics.items():
        if value!=99:
          cv_metrics[key].append(value)

  avg_metrics = OrderedDict([('acc', np.mean(cv_metrics['acc'])), ('N_acc', np.mean(cv_metrics['N_acc'],0)),('C_acc', np.mean(cv_metrics['C_acc'],0)),
                               ('A_acc', np.mean(cv_metrics['A_acc'],0)),('P_acc', np.mean(cv_metrics['P_acc'],0)),('CA_acc', np.mean(cv_metrics['CA_acc'],0)),
                               ('CP_acc', np.mean(cv_metrics['CP_acc'],0)),('AP_acc', np.mean(cv_metrics['AP_acc'],0)),('CAP_acc', np.mean(cv_metrics['CAP_acc'],0))])
  update_cv('avg_metrics',roundOrder(avg_metrics),  os.path.join(foldZeroOutputDir, 'summary.csv'), write_header=True)
  
  std_metrics =  OrderedDict([('acc', np.std(cv_metrics['acc'])),('N_acc', np.std(cv_metrics['N_acc'],0)),('C_acc', np.std(cv_metrics['C_acc'],0)),
                               ('A_acc', np.std(cv_metrics['A_acc'],0)),('P_acc', np.std(cv_metrics['P_acc'],0)),('CA_acc', np.std(cv_metrics['CA_acc'],0)),
                               ('CP_acc', np.std(cv_metrics['CP_acc'],0)),('AP_acc', np.std(cv_metrics['AP_acc'],0)),('CAP_acc', np.std(cv_metrics['CAP_acc'],0))])
  update_cv('std_metrics',roundOrder(std_metrics), os.path.join(foldZeroOutputDir, 'summary.csv'), write_header=True)
  
if __name__ == '__main__':
    main()



