import torch

# Set seed
torch.manual_seed(1)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataset import *
from functools import partial
from backbone import *
from rpn_new import *
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

  backbone = Resnet50Backbone()
  rpn_net = RPNHead()
  return backbone, rpn_net, train_loader, test_loader

###########################################
# Train function
###########################################
def train(backbone, rpn_net, train_loader, test_loader, resume_checkpoint = None):
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

  # Hyperparameters
  learning_rate = 0.005
  weight_decay = 0.0
  num_epochs = 40

  ## Intialize Optimizer
  optimizer=torch.optim.Adam(rpn_net.parameters(), lr=learning_rate, weight_decay=weight_decay)

  ## Keep Track of Losses
  losses = []
  class_losses = []
  reg_losses = []

  if resume_checkpoint:
    checkpoint = torch.load(
        os.path.join(path, resume_checkpoint),
        map_location=device
    )
    rpn_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']+1
    losses = checkpoint['losses']
    class_losses = checkpoint['class_losses']
    reg_losses = checkpoint['reg_losses']
  else:
    epoch = 0
  
  # Load the ground truth
  if not hasattr(rpn_net, 'ground_dict'):
    gt_cache_path = os.path.join(path, 'rpn_ground_truth_cache')
    if os.path.exists(gt_cache_path):
      rpn_net.ground_dict = torch.load(gt_cache_path, map_location=device)

  for epoch in range(epoch, num_epochs):
    # Ready the network for training
    rpn_net.train()
    
    # Intialize list to hold running losses during batch training
    running_losses = []
    running_class_losses = []
    running_reg_losses = []


    for batch_idx, data in enumerate(train_loader):
        # Get raw data
        images=data['images'].to(device)
        len_images = len(images)
        indexes=data['index']
        boxes=[bbox.to(device) for bbox in data['bbox']]
        targ_clas,targ_regr=rpn_net.create_batch_truth(boxes,indexes,[images.shape[-2:]]*len(images))

        # Get the predictions by doing a forward pass
        optimizer.zero_grad()
        backout = backbone(images)
        X = [backout["0"], backout["1"], backout["2"], backout["3"], backout["pool"]]
        clas_out,regr_out = rpn_net.forward(X)


        # Calculate the loss
        del images, indexes, boxes, backout, X
        if torch.cuda.is_available():
          torch.cuda.empty_cache()
        loss, class_loss, reg_loss = rpn_net.compute_loss(
          clas_out,
          regr_out,
          targ_clas,
          targ_regr,
          l=.1,
          effective_batch=[10, 10, 10, 10, 10] # 5 different effective batches for each FPN level
        )

        # Save the losses for later
        running_losses.append(loss.item())
        running_class_losses.append(class_loss.item())
        running_reg_losses.append(reg_loss.item())

        # Backprop
        del targ_clas, targ_regr, clas_out, regr_out, class_loss, reg_loss
        if torch.cuda.is_available():
          torch.cuda.empty_cache()
        loss.backward()
        optimizer.step()

        # Print batch status
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len_images, len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))

        del loss

    # After all batches add losses to the curves 
    losses.extend(running_losses)
    class_losses.extend(running_class_losses)
    reg_losses.extend(running_reg_losses)

    # Print epoch status
    print(
        "Epoch:", epoch,
        "Classification Loss:", sum(running_class_losses) / float(len(running_class_losses)),
        "Regression Loss:", sum(running_reg_losses) / float(len(running_reg_losses)),
        "Total Loss:", sum(running_losses) / float(len(running_losses)),
    )

    # Save a checkpoint at the end of each epoch
    chkpt_path = os.path.join(path,'rpn_epoch_'+str(epoch))
    torch.save({
      'epoch': epoch,
      'model_state_dict': rpn_net.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'losses': losses,
      'class_losses': class_losses,
      'reg_losses': reg_losses,
    }, chkpt_path)

    # Save the ground truth
    if epoch == 0:
      gt_cache_path = os.path.join(path, 'rpn_ground_truth_cache')
      torch.save(rpn_net.ground_dict, gt_cache_path)

  return losses, class_losses, reg_losses

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

def plot_loss_curves(losses, class_losses, reg_losses):
    # Setup the plot
    plt.figure(figsize=(7,15))

    # Average Losses over 100 iterations
    new_losses = []
    new_class_losses = []
    new_reg_losses = []
    for i in range(int(len(losses)/100)):
      new_losses.append(sum(losses[i*100:(i+1)*100])/100)
      new_class_losses.append(sum(class_losses[i*100:(i+1)*100])/100)
      new_reg_losses.append(sum(reg_losses[i*100:(i+1)*100])/100)

    # Plot the loss curves
    ax = plt.subplot(3, 1, 1)
    plot_array(ax, "Training Total Loss", "Iteration", "Total Loss", new_losses)
    ax = plt.subplot(3, 1, 2)
    plot_array(ax, "Training Classification Loss", "Iteration", "Classification Loss", new_class_losses)
    ax = plt.subplot(3, 1, 3)
    plot_array(ax, "Training Regression Loss", "Iteration", "Regression Loss", new_reg_losses)

    # Show the plot
    plt.show()

###########################################
# Main
###########################################
if __name__ == '__main__':
    print("Building the dataset...")
    backbone, rpn_net, train_loader, test_loader = build_dataset()
    
    print("Training...")
    losses, class_losses, reg_losses = train(backbone, rpn_net, train_loader, test_loader)
    
    # Plot the loss curves
    print("Plotting loss curves...")
    plot_loss_curves(losses, class_losses, reg_losses)

