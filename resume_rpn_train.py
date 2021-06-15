from main_rpn_train import *

if __name__ == '__main__':
    print("Building the dataset...")
    backbone, rpn_net, train_loader, test_loader = build_dataset()
    
    # In the last argument of train() put the name of the 
    # checkpoint file you want to resume training from
    print("Training...")
    losses, class_losses, reg_losses = train(backbone, rpn_net, train_loader, test_loader, 'rpn_epoch_0')
    
    # Plot the loss curves
    print("Plotting loss curves...")
    plot_loss_curves(losses, class_losses, reg_losses)