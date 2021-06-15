  
import torch

# Set seed
torch.manual_seed(1)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataset import *
from functools import partial
from pretrained_models import *
from BoxHead import *
from main_train import *
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
# P/R Curve function
###########################################
def calculate_pr_info(total_positives, running_pr_info, scores, labels, boxes, gt_labels, decoded_coord):
  flat_list_of_categories = gt_labels[0].flatten().tolist() # [0] since we are doing batch_size = 1
  total_positives[0] += sum([category == 1 for category in flat_list_of_categories])
  total_positives[1] += sum([category == 2 for category in flat_list_of_categories])
  total_positives[2] += sum([category == 3 for category in flat_list_of_categories])

  # Go through each prediction
  for confidence_score, category_pred, coord_pred in zip(scores, labels, boxes):
    coord_pred = coord_pred.flatten().tolist()
    predicted_correct = False

    # Check if any of the ground truth boxes match the prediction
    for idx, category_gt in enumerate(flat_list_of_categories):
      coord_gt = decoded_coord[idx].cpu().data.numpy().flatten().tolist()
      class_correct = int(category_gt) == int(category_pred)
      iou = IOU(coord_pred, coord_gt)
      iou_correct = iou >= .5
      if class_correct and iou_correct:
        predicted_correct = True

    # Record the information for the prediction
    if int(category_pred) == 1:
      running_pr_info[0].append((confidence_score, predicted_correct))
    if int(category_pred) == 2:
      running_pr_info[1].append((confidence_score, predicted_correct))
    if int(category_pred) == 3:
      running_pr_info[2].append((confidence_score, predicted_correct))


def average_precision(cls, total_positives, running_pr_info, return_curves = False):
  running_pr_info[cls].sort(reverse=True)
  
  correct = 0
  precisions = []
  recalls = []
  total_positives = [cls_total_positives.item() for cls_total_positives in total_positives]

  for i in range(len(running_pr_info[cls])):
    if running_pr_info[cls][i][1]:
      correct += 1
    precisions.append(correct / (i+1))
    if total_positives[cls] == 0:
        recalls.append(0)
    else:   
        recalls.append(correct / total_positives[cls])

  # print("Precisions:", precisions[0:500])
  # print("Recalls:", recalls[0:500])

  if not return_curves:
    return auc(recalls, precisions)
  else:
    return precisions, recalls, auc(recalls, precisions)

def mean_average_precision(total_positives, running_pr_info):
  ap0 = average_precision(0, total_positives, running_pr_info)
  ap1 = average_precision(1, total_positives, running_pr_info)
  ap2 = average_precision(2, total_positives, running_pr_info)
  print("Average Precisions of Classes:", ap0, ap1, ap2)
  return (ap0+ap1+ap2)/3.0


###########################################
# Plot function
###########################################
def PlotInfer(title, boxes, labels, masks, img, idx, batch_idx):
  plot_img = img
  plot_img = transforms.functional.normalize(plot_img,(-0.485/0.229,-0.456/0.224,-0.406/0.225),(1/0.229,1/0.224,1/0.225))
  plot_img = plot_img.permute(1,2,0).cpu()

  fig,ax=plt.subplots(1,1)
  ax.imshow(plot_img)
  ax.set_title("Image "+str(idx)+" in Batch "+str(batch_idx)+" -- " + title)

  for idx, coord in enumerate(boxes):
    coord = coord.flatten().tolist()
    color_list = ["m", "r", "g"]
    label = labels[idx]
    color = color_list[int(label.item()) - 1]
    rect=patches.Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],fill=False,color=color)
    ax.add_patch(rect)

  plt.show()
##########################################################################
#This function overlays the projected masks and
#creates a refined mask which is used for plotting masks onto the image
##########################################################################
def PlotMask(images,projected_masks,labels):
  
    color_list = ["jet", "ocean", "Spectral"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    plot_img = transforms.functional.normalize(images.to('cpu'),(-0.485/0.229,-0.456/0.224,-0.406/0.225),(1/0.229,1/0.224,1/0.225)) 
    plot_img = plot_img.permute(1,2,0)
    fig,ax=plt.subplots(1,1)
    ax.imshow(plot_img)
    refined_masks = torch.zeros(3,800,1088).to(device)

    k = torch.zeros(3)
    m = torch.zeros(3)

    for idx,labs in enumerate(labels):
      refined_masks[int(labs.item()) - 1] = refined_masks[int(labs.item()) - 1] + projected_masks[idx].to(device)
      k[int(labs.item()) - 1] +=1

    for idx in range(len(k)):
      if k[idx]>=5:
        m[idx] = k[idx]/5
      else:
        m[idx] = 1

    # refined_masks = refined_masks/k

    for idx,ref_mask in enumerate(refined_masks):
      color = color_list[idx]
      ref_mask = ref_mask.clone()
      ref_mask = ref_mask/m[idx]
      # refined_masks[idx] = torch.clamp(refined_masks[idx],0,1)
      refined_masks[idx][refined_masks[idx]<0.7]=0
      refined_masks[idx][refined_masks[idx]>=0.7]=1
      mask_new = np.ma.masked_where(refined_masks[idx].cpu().data.numpy() == 0, refined_masks[idx].cpu().data.numpy())
      plt.imshow(mask_new, cmap=color, alpha=0.7)

    plt.show()

###########################################
# Test function
###########################################
def test(backbone, rpn, box_head_net, mask_head_net, train_loader, test_loader, resume_checkpoint):
  from torchvision.models.detection.image_list import ImageList
  
  # Create folder for checkpoints
  if IN_COLAB:
    path = os.path.join(HOMEWORK_FOLDER, 'checkpoints')
  else:
    path = os.path.join('.', 'checkpoints')
  os.makedirs(path, exist_ok=True)
  
  # Get the device
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # Initialize network
  mask_head_net=mask_head_net.to(device)

  # Hyperparameters
  keep_topK=100

  checkpoint = torch.load(
      os.path.join(path, resume_checkpoint),
      map_location=device
  )
  mask_head_net.load_state_dict(checkpoint['model_state_dict'])

  # Load the ground truth
  # TODO: Uncomment later
  # if not hasattr(mask_head_net, 'ground_dict'):
  #   gt_cache_path = os.path.join(path, 'mask_head_ground_truth_cache')
  #   if os.path.exists(gt_cache_path):
  #     mask_head_net.ground_dict = torch.load(gt_cache_path, map_location=device)

  # Ready the network for training
  mask_head_net.eval()

  # P/R variables
  mean_avg_precisions = []
  total_positives = [torch.zeros((1,)).to(device), torch.zeros((1,)).to(device), torch.zeros((1,)).to(device)]
  running_pr_info = [[], [], []]

  for batch_idx, data in enumerate(test_loader):
    # Get raw data
    images=data['images'].to(device)
    len_images = len(images)
    indexes=data['index']
    gt_labels=data['labels']
    gt_boxes=[bbox.to(device) for bbox in data['bbox']]
    masks=data['masks'].to(device)

    # Take the features from the backbone
    backout = backbone(images)
    im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
    rpnout = rpn(im_lis, backout)
    proposals=[crop_to_image_size(proposal[0:keep_topK,:]) for proposal in rpnout[0]]
    fpn_feat_list= list(backout.values())
    with torch.no_grad():
      feature_vectors = box_head.MultiScaleRoiAlign(fpn_feat_list,proposals,P=box_head.P)
      class_logits, box_preds = box_head.forward(feature_vectors)
      del feature_vectors

    # Create the ground truth
    boxes, scores, labels, gt_masks = mask_head_net.preprocess_ground_truth_creation(
      class_logits,
      box_preds,
      gt_labels,
      gt_boxes,
      masks
    )

    # Get the predictions by doing a forward pass
    feature_vectors = MultiScaleRoiAlign(fpn_feat_list, boxes, P = 14)
    mask_outputs = mask_head_net.forward(feature_vectors)


    # Do whatever post processing you find performs best
    projected_masks = mask_head_net.postprocess_mask(masks_outputs, boxes, gt_labels)


    # Plot the inference images as we go
    print("Running plotting on batch...")
    PlotInfer(
      "Post-NMS Prediction",
      boxes[0], # We can just do 0 here since batch_size =1
      labels[0], # We can just do 0 here since batch_size =1
      projected_masks[0], # We can just do 0 here since batch_size =1
      images[0], # We can just do 0 here since batch_size =1
      0, # We can just do 0 here since batch_size =1
      batch_idx
    )
    
    ## Uncomment the following line to plot the images with refined masks
    #PlotMask(images[0],projected_masks[0],labels[0])

    # Calculate PR info
    # _, regressor_target = box_head_net.create_ground_truth(proposals, gt_labels, gt_boxes, indexes)
    # # Flatten the proposals and convert them to cx, xy, w, h format
    # flattened_proposals = torch.stack(proposals, dim=0).reshape((regressor_target.shape))
    # flattened_proposals_width = (flattened_proposals[:, 2] - flattened_proposals[:, 0])
    # flattened_proposals_height = (flattened_proposals[:, 3] - flattened_proposals[:, 1])
    # flattened_proposals_center_x = (flattened_proposals[:, 0] + flattened_proposals_width/2)
    # flattened_proposals_center_y = (flattened_proposals[:, 1] + flattened_proposals_height/2)
    # flattened_proposals[:, 0] = flattened_proposals_center_x
    # flattened_proposals[:, 1] = flattened_proposals_center_y
    # flattened_proposals[:, 2] = flattened_proposals_width
    # flattened_proposals[:, 3] = flattened_proposals_height
    # decoded_coord = output_decoding(regressor_target, flattened_proposals)
    # calculate_pr_info(total_positives, running_pr_info, scores[0], labels[0], boxes[0], projected_masks[0], gt_labels, decoded_coord.cpu())

    # Print batch status
    print('Test: [{}/{} ({:.0f}%)]'.format(
      batch_idx * len(images), len(test_loader.dataset),
      100. * batch_idx / len(test_loader)))
    
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

  # precisions0, recalls_0, auc_0 = average_precision(0, total_positives, running_pr_info, return_curves = True)
  # precisions1, recalls_1, auc_1 = average_precision(1, total_positives, running_pr_info, return_curves = True)
  # precisions2, recalls_2, auc_2 = average_precision(2, total_positives, running_pr_info, return_curves = True)
  # mAP = mean_average_precision(total_positives, running_pr_info)

  # Print final status
  print(
      "Finished running inference on all test images"
  )

  # Show P/R information & plots

  # Setup the plot
  plt.figure(figsize=(7,15))
  print("Mean Average Precision over Test Inference:", mAP)
  ax = plt.subplot(3, 1, 1)
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  plot_array(ax, "P/R Curve (Class = 1)", "Recall", "Precision", precisions0, recalls_0)
  ax = plt.subplot(3, 1, 2)
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  plot_array(ax, "P/R Curve (Class = 2)", "Recall", "Precision", precisions1, recalls_1)
  ax = plt.subplot(3, 1, 3)
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  plot_array(ax, "P/R Curve (Class = 3)", "Recall", "Precision", precisions2, recalls_2)

  plt.show()

###########################################
# Main
###########################################
if __name__ == '__main__':

    print("Building the dataset...")
    backbone, rpn, box_head_net, mask_head_net, train_loader, test_loader = build_dataset()
    
    # In the last argument of test() put the name of the 
    # checkpoint file you want to load the model from
    print("Running inference on the model...")
    test(backbone, rpn, box_head_net, mask_head_net, train_loader, test_loader, 'maskhead_epoch_39')
