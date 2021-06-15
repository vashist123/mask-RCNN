import torch

# Set seed
torch.manual_seed(1)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from dataset import *
from functools import partial
from rpn_new import *
from main_rpn_train import *
from sklearn.metrics import auc
import os
import time
import itertools


#################
# Detect CoLab
#################
try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

###########################################
# Plot function
###########################################
def PlotInfer(title, clas, prebox, img, idx, batch_idx):
  plot_img = img
  plot_img = transforms.functional.normalize(plot_img,(-0.485/0.229,-0.456/0.224,-0.406/0.225),(1/0.229,1/0.224,1/0.225))
  plot_img = plot_img.permute(1,2,0).cpu()

  fig,ax=plt.subplots(1,1)
  ax.imshow(plot_img)
  ax.set_title("Image "+str(idx)+" in Batch "+str(batch_idx)+" -- " + title)

  for coord in prebox:
    coord = coord.cpu().data.numpy().flatten().tolist()
    col='r'
    rect=patches.Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],fill=False,color=col)
    ax.add_patch(rect)

  plt.show()

###########################################
# Test function
###########################################
def test(backbone, rpn_net, train_loader, test_loader, resume_checkpoint):
  # Create folder for checkpoints
  if IN_COLAB:
    path = os.path.join(HOMEWORK_FOLDER, 'checkpoints')
  else:
    path = os.path.join('.', 'checkpoints')
  os.makedirs(path, exist_ok=True)
  
  # Get the device
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # Initialize network
  rpn_net=rpn_net.to(device)

  # Load the checkpoint
  checkpoint = torch.load(
      os.path.join(path, resume_checkpoint),
      map_location=device
  )
  rpn_net.load_state_dict(checkpoint['model_state_dict'])
  

  # Ready the network for inference
  rpn_net.eval()

  # Variables for point-wise accuracy
  correct = 0
  total = 0

  for batch_idx, data in enumerate(test_loader):
    # Get raw data
    images=data['images'].to(device)
    len_images = len(images)
    indexes=data['index']
    boxes=[bbox.to(device) for bbox in data['bbox']]
    targ_clas,targ_regr=rpn_net.create_batch_truth(boxes,indexes,[images.shape[-2:]]*len(images))

    # Get the predictions by doing a forward pass
    backout = backbone(images)
    X = [backout["0"], backout["1"], backout["2"], backout["3"], backout["pool"]]
    clas_out,regr_out = rpn_net.forward(X)

    # Compute point-wise accuracy
    # Computed according to: https://piazza.com/class/kedjeyrln6icm?cid=299
    # Next line: Flattens the predictions and thresholds them to 0 or 1 based <.5 or >.5 using round()
    flat_clas_out = [min(max(round(x), 0), 1) for x in itertools.chain.from_iterable([c.cpu().data.numpy().flatten().tolist() for c in clas_out])]
    # Next line: Flattens target classes, treats -1 class as the 0 class as mentioned in Piazza post
    flat_targ_clas = [max(x, 0) for x in itertools.chain.from_iterable([c.cpu().data.numpy().flatten().tolist() for c in targ_clas])]
    total += len(flat_targ_clas)
    # https://www.kite.com/python/answers/how-to-count-the-number-of-true-booleans-in-a-list-in-python#:~:text=Use%20sum()%20to%20count,True%20booleans%20in%20the%20list.
    # Counts how many we got right
    correct += sum([predict_class == target_class for predict_class, target_class in zip(flat_clas_out, flat_targ_clas)])


    # Print batch status
    print('Test: [{}/{} ({:.0f}%)]'.format(
    batch_idx * len(images), len(test_loader.dataset),
    100. * batch_idx / len(test_loader)))

    # Post process the predictions
    print("Running post-processing on batch...")
    clas_list, prebox_list, nms_clas_list, nms_prebox_list  = rpn_net.postprocess(
      clas_out,
      regr_out,
      IOU_thresh=0.5,
      keep_num_preNMS=50,
      keep_num_postNMS=10
    )

    # Plot the inference images as we go
    print("Running plotting on batch...")
    for idx, img in enumerate(images):
      PlotInfer(
        "Top 20 Proposals",
        clas_list[idx][:20],
        prebox_list[idx][:20],
        img,
        idx,
        batch_idx
      )
      PlotInfer(
        "Pre-NMS",
        clas_list[idx],
        prebox_list[idx],
        img,
        idx,
        batch_idx
      )
      PlotInfer(
        "Post-NMS",
        nms_clas_list[idx],
        nms_prebox_list[idx],
        img,
        idx,
        batch_idx
      )


  # Print final status
  print("Point-wise Accuracy:", (correct/total))
  print(
      "Finished running inference on all test images"
  )



###########################################
# Main
###########################################
if __name__ == '__main__':
    print("Building the dataset...")
    backbone, rpn_net, train_loader, test_loader = build_dataset()
    
    # In the last argument of test() put the name of the 
    # checkpoint file you want to load the model from
    print("Running inference on the model...")
    test(backbone, rpn_net, train_loader, test_loader, 'rpn_epoch_19')