from main_train import *

if __name__ == '__main__':
    print("Building the dataset...")
    backbone, rpn, box_head_net, mask_head_net, train_loader, test_loader = build_dataset()
    
    # In the last argument of train() put the name of the 
    # checkpoint file you want to resume training from
    print("Training...")
    losses, validation_losses = train(backbone, rpn, box_head_net, mask_head_net, train_loader, test_loader, 'maskhead_epoch_39')
    
    # Plot the loss curves
    print("Plotting loss curves...")
    plot_loss_curves("Training", "Iteration", losses)
    plot_loss_curves("Validation", "Epoch", validation_losses, 1)