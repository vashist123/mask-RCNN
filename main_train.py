  
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
from MaskHead import *
import os
import time


#################
# Detect CoLab
#################
try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

###########################################
# Build dataset
###########################################
def build_dataset():
  if IN_COLAB:
      path = os.path.join(HOMEWORK_FOLDER, 'checkpoints')
      pretrained_path=os.path.join(HOMEWORK_FOLDER, "checkpoints/checkpoint680.pth")
  else:
      path = os.path.join('.', 'checkpoints')
      pretrained_path='checkpoints/checkpoint680.pth'
  if IN_COLAB:
      imgs_path = os.path.join(HOMEWORK_FOLDER, "data/hw3_mycocodata_img_comp_zlib.h5")
      masks_path = os.path.join(HOMEWORK_FOLDER, "data/hw3_mycocodata_mask_comp_zlib.h5")
      labels_path = os.path.join(HOMEWORK_FOLDER, "data/hw3_mycocodata_labels_comp_zlib.npy")
      bboxes_path = os.path.join(HOMEWORK_FOLDER, "data/hw3_mycocodata_bboxes_comp_zlib.npy")
  else:
      imgs_path = os.path.join('.', "data/hw3_mycocodata_img_comp_zlib.h5")
      masks_path = os.path.join('.', "data/hw3_mycocodata_mask_comp_zlib.h5")
      labels_path = os.path.join('.', "data/hw3_mycocodata_labels_comp_zlib.npy")
      bboxes_path = os.path.join('.', "data/hw3_mycocodata_bboxes_comp_zlib.npy")
  paths = [imgs_path, masks_path, labels_path, bboxes_path]
  # load the data into data.Dataset
  dataset = BuildDataset(paths)

  # --------------------------------------------
  # build the dataloader
  # set 20% of the dataset as the training data
  full_size = len(dataset)
  train_size = int(full_size * 0.8)
  test_size = full_size - train_size
  # random split the dataset into training and testset
  # set seed
  torch.random.manual_seed(1)
  train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
  # push the randomized training data into the dataloader

  # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
  # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
  batch_size = 4
  train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
  train_loader = train_build_loader.loader()
  test_build_loader = BuildDataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
  test_loader = test_build_loader.loader()

  backbone, rpn = pretrained_models_680(pretrained_path)
  box_head_net = BoxHead()
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  box_head_net=box_head_net.to(device)
  # box_head_net.eval() TODO: uncomment later
  box_head_checkpoint = torch.load(
      os.path.join(path, 'boxhead2_epoch_39'),
      map_location=device
  )
  box_head_net.load_state_dict(box_head_checkpoint['model_state_dict'])
  # Load the ground truth
  if not hasattr(box_head_net, 'ground_dict'):
    gt_cache_path = os.path.join(path, 'ground_truth_cache')
    if os.path.exists(gt_cache_path):
      box_head_net.ground_dict = torch.load(gt_cache_path, map_location=device)
  mask_head_net = MaskHead()
  return backbone, rpn, box_head_net, mask_head_net, train_loader, test_loader

###########################################
# Train function
###########################################
def train(backbone, rpn, box_head_net, mask_head_net, train_loader, test_loader, resume_checkpoint = None):
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
  learning_rate = 0.001
  weight_decay = 0.0005
  num_epochs = 40

  ## Intialize Optimizer
  optimizer=torch.optim.Adam(mask_head_net.parameters(), lr=learning_rate, weight_decay=weight_decay)

  ## Keep Track of Losses
  losses = []
  validation_losses = []

  if resume_checkpoint:
    checkpoint = torch.load(
        os.path.join(path, resume_checkpoint),
        map_location=device
    )
    mask_head_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']+1
    losses = checkpoint['losses']
    validation_losses = checkpoint['validation_losses']
  else:
    epoch = 0

  # Load the ground truth
  # TODO: Uncomment later
  # if not hasattr(mask_head_net, 'ground_dict'):
  #   gt_cache_path = os.path.join(path, 'mask_head_ground_truth_cache')
  #   if os.path.exists(gt_cache_path):
  #     mask_head_net.ground_dict = torch.load(gt_cache_path, map_location=device)
  
  for epoch in range(epoch, num_epochs):
    # Ready the network for training
    mask_head_net.train()
    
    # Intialize list to hold running losses during batch training
    running_losses = []


    for batch_idx, data in enumerate(train_loader):
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
          feature_vectors = box_head_net.MultiScaleRoiAlign(fpn_feat_list,proposals,P=box_head_net.P)
          class_logits, box_preds = box_head_net.forward(feature_vectors)
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
        optimizer.zero_grad()
        feature_vectors = MultiScaleRoiAlign(fpn_feat_list, boxes, P=14)
        mask_outputs = mask_head_net.forward(feature_vectors)


        # Calculate the loss
        loss, = mask_head_net.compute_loss(
          mask_outputs,
          labels,
          gt_masks
        )

        # Backprop
        loss.backward()
        optimizer.step()

        # Save the losses for later
        running_losses.append(loss.item())

        # Print batch status
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(images), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))

    # After all batches add losses to the curves 
    losses.extend(running_losses)


    ############################################
    # BEGIN VALIDATION CURVE CODE 
    # Comment this section out to make it faster, if 
    # you don't want validation curves
    ############################################
    running_validation_losses = []

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
          feature_vectors = box_head_net.MultiScaleRoiAlign(fpn_feat_list,proposals,P=box_head_net.P)
          class_logits, box_preds = box_head_net.forward(feature_vectors)
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
        optimizer.zero_grad()
        feature_vectors = MultiScaleRoiAlign(fpn_feat_list, boxes, P=14)
        mask_outputs = mask_head_net.forward(feature_vectors)


        # Calculate the loss
        loss, = mask_head_net.compute_loss(
          mask_outputs,
          labels,
          gt_masks
        )

        # Save the losses for later
        running_validation_losses.append(loss.item())

        # Print batch status
        print('Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(images), len(test_loader.dataset),
          100. * batch_idx / len(test_loader), loss.item()))

    # After all batches add losses to the curves 
    validation_losses.append(sum(running_validation_losses) / float(len(running_validation_losses)))
    ############################################
    # END VALIDATION CURVE CODE
    ############################################

    # Print epoch status
    print(
        "Epoch:", epoch,
        "Total Loss:", sum(running_losses) / float(len(running_losses)),
    )
    print(
        "Epoch:", epoch,
        "Validation Total Loss:", validation_losses[-1:],
    )

    # Save a checkpoint at the end of each epoch
    chkpt_path = os.path.join(path,'maskhead_epoch_'+str(epoch))
    torch.save({
      'epoch': epoch,
      'model_state_dict': mask_head_net.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'losses': losses,
      'validation_losses': validation_losses,
    }, chkpt_path)

    # Save the ground truth
    # TODO: Uncomment later
    # if epoch == 0:
    #   gt_cache_path = os.path.join(path, 'mask_head_ground_truth_cache')
    #   torch.save(mask_head_net.ground_dict, gt_cache_path)

  return losses, validation_losses

###########################################
# Plotting Loss Curves function
###########################################

def plot_array(ax, title, xlabel, ylabel, y_data, x_data = None):
    if x_data is None:
        ax.plot(y_data)
    else:
        ax.plot(x_data, y_data)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def plot_loss_curves(title, axis_title, losses, window = 100):
    # Setup the plot
    plt.figure(figsize=(10,10))

    # Average Losses over 100 iterations
    new_losses = []
    for i in range(int(len(losses)/window)):
      new_losses.append(sum(losses[i*window:(i+1)*window])/window)

    # Plot the loss curves
    ax = plt.subplot(1, 1, 1)
    plot_array(ax, title + " Total Loss", axis_title, "Total Loss", new_losses)

    # Show the plot
    plt.show()

###########################################
# Main
###########################################
if __name__ == '__main__':
    print("Building the dataset...")
    backbone, rpn, box_head_net, mask_head_net, train_loader, test_loader = build_dataset()
    
    print("Training...")
    losses, validation_losses = train(backbone, rpn, box_head_net, mask_head_net, train_loader, test_loader)
    
    # Plot the loss curves
    print("Plotting loss curves...")
    plot_loss_curves("Training", "Iteration", losses)
    plot_loss_curves("Validation", "Epoch", validation_losses, 1)