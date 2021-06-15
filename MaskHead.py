import torch
import torch.nn.functional as F
from torch import nn
from utils import *

class MaskHead(torch.nn.Module):
    def __init__(self,Classes=3,P=14):
        self.C=Classes
        self.P=P
        # TODO initialize MaskHead
        self.MaskBranch = torch.nn.Sequential(
        	torch.nn.Conv2d(256,256,kernel_size=3,padding=1),
        	torch.nn.ReLU(),
        	torch.nn.Conv2d(256,256,kernel_size=3,padding=1),
        	torch.nn.ReLU(),
        	torch.nn.Conv2d(256,256,kernel_size=3,padding=1),
        	torch.nn.ReLU(),
        	torch.nn.Conv2d(256,256,kernel_size=3,padding=1),
        	torch.nn.ReLU(),
        	torch.nn.ConvTranspose2d(256,256,kernel_size=8).
        	torch.nn.ReLU(),
        	torch.nn.Conv2d(256,3,kernel_size=1),
        	torch.nn.Sigmoid()
        	)


    # This function does the pre-prossesing of the proposals created by the Box Head (during the training of the Mask Head)
    # and create the ground truth for the Mask Head
    #
    # Input:
    #       proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #       class_logits: (total_proposals,(C+1))
    #       box_regression: (total_proposal,4*C)  ([t_x,t_y,t_w,t_h])
    #       gt_labels: list:len(bz) {(n_obj)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #       masks: list:len(bz){(n_obj,800,1088)}
    #       IOU_thresh: scalar (threshold to filter regressed with low IOU with a bounding box)
    #       keep_num_preNMS: scalar (number of boxes to keep pre NMS)
    #       keep_num_postNMS: scalar (number of boxes to keep post NMS)
    # Output:
    #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)} ([x1,y1,x2,y2] format)
    #       scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
    #       labels: list:len(bz){(post_NMS_boxes_per_image)}  (top category of each regressed box)
    #       gt_masks: list:len(bz){(post_NMS_boxes_per_image,2*P,2*P)}
    #       props   : list:len(bz){(post_NMS_boxes_per_image,4)} ([x1,y1,x2,y2] format)
    def preprocess_ground_truth_creation(self, proposals, class_logits, box_regression, gt_labels,bbox ,masks , IOU_thresh=0.5, keep_num_preNMS=1000, keep_num_postNMS=100):
      bz = len(proposals)
      per_image_proposals = proposals[0].shape[0]
      total_proposals = class_logits.shape[0]

      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

      assert sum([p.shape[0] for p in proposals]) == len(box_regression) # Makes sure that total_proposals in box_regression is the same number of total proposals in the "proposals" variable
      assert all([p.shape[0] == per_image_proposals for p in proposals]) # Make sure that each proposal in proposals has length 200
      boxes = []
      scores = []
      labels = []
      gt_masks=[]
      props = []
      
      for img_idx in range(bz):
        img_class_logits = class_logits[img_idx*per_image_proposals:(img_idx+1)*per_image_proposals]
        img_box_regression = box_regression[img_idx*per_image_proposals:(img_idx+1)*per_image_proposals]
        
        # try:
        img_boxes,img_scores,img_labels,img_gt_masks,propo = self.image_preprocess_ground_truth_creation(proposals[img_idx],img_class_logits,img_box_regression,gt_labels[img_idx],bbox[img_idx],masks[img_idx],0.5,1000,100)
        # except:
        #   print("No boxes for image at index: ", img_idx)
        #   img_boxes,img_scores,img_labels,img_gt_masks,propo = torch.zeros((0, 4)).to(device), torch.zeros((0,)).to(device), torch.zeros((0,)).to(device), torch.zeros((0, 28, 28)).to(device), torch.zeros((0, 4)).to(device)

        boxes.append(img_boxes)
        scores.append(img_scores)
        labels.append(img_labels)
        gt_masks.append(img_gt_masks)
        props.append(propo)
      
      return boxes, scores, labels, gt_masks, props


    def image_preprocess_ground_truth_creation(self,proposals_img,class_logits_img,box_regression_img,gt_labels_img,bbox_img,masks_img,IOU_thresh=0.5, keep_num_preNMS=1000, keep_num_postNMS=100):

      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      
      scores,classes = torch.max(class_logits_img,dim=1)
      zero_idx = torch.nonzero(classes[:]==0)
      scores[zero_idx[:]] = 0
      tresh_scores_ind = torch.nonzero(scores[:]>0.5)
      img_scores = scores[tresh_scores_ind[:]].squeeze()
      img_labels = classes[tresh_scores_ind[:]].squeeze()
      new_proposals = proposals_img[tresh_scores_ind[:]].squeeze()
      new_box_regression = box_regression_img[tresh_scores_ind[:]].squeeze()
      img_pre_boxes = torch.zeros(new_box_regression.shape[0],4).to(device)
      
      for idx,i in enumerate(new_box_regression):
        try:
          lab = (img_labels[idx]-1)*4
          img_pre_boxes[idx][0:4] = i[lab:lab+4]
        except:
          img_pre_boxes[idx][0:4] = torch.zeros(1,4)
        
        
      img_boxes = output_decodingd(img_pre_boxes,new_proposals)
      img_gt_masks = torch.zeros(img_boxes.shape[0],28,28).to(device)
      for ref_box_idx in range(len(img_boxes)):
        img_boxes[ref_box_idx][0] = torch.clamp(img_boxes[ref_box_idx][0],0,800)
        img_boxes[ref_box_idx][1] = torch.clamp(img_boxes[ref_box_idx][1],0,1088)
        img_boxes[ref_box_idx][2] = torch.clamp(img_boxes[ref_box_idx][2],0,800)
        img_boxes[ref_box_idx][3] = torch.clamp(img_boxes[ref_box_idx][3],0,1088)
        max_iou = None
        max_gt_mask = None
        max_idx = None
        for gt_idx in range(len(bbox_img)):
          iou = IOU(bbox_img[gt_idx].cpu().data.numpy().flatten().tolist(),img_boxes[ref_box_idx].cpu().data.numpy().flatten().tolist())
          if iou>0.01 and img_labels[ref_box_idx]==gt_labels_img[gt_idx]:
            if max_iou is None or iou> max_iou:
              max_iou = iou
              max_gt_mask = masks_img[gt_idx]
              max_idx = gt_idx
        if max_iou is not None:

          int_mask = max_gt_mask[int(img_boxes[ref_box_idx][0]):int(img_boxes[ref_box_idx][2]),int(img_boxes[ref_box_idx][1]):int(img_boxes[ref_box_idx][3])]

          img_gt_masks[ref_box_idx] = torch.nn.functional.interpolate(int_mask.unsqueeze(0).unsqueeze(0),size = (28,28),mode ='bilinear').squeeze(0).squeeze(0)
      
      return img_boxes,img_scores,img_labels,img_gt_masks,new_proposals

    # general function that takes the input list of tensors and concatenates them along the first tensor dimension
    # Input:
    #      input_list: list:len(bz){(dim1,?)}
    # Output:
    #      output_tensor: (sum_of_dim1,?)
    def flatten_inputs(self,input_list):

      output_tensor = torch.cat(input_list,dim=0)

      return output_tensor

    # This function does the post processing for the result of the Mask Head for a batch of images. It project the predicted mask
    # back to the original image size
    # Use the regressed boxes to distinguish between the images
    # Input:
    #       masks_outputs: (total_boxes,C,2*P,2*P)
    #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)} ([x1,y1,x2,y2] format)
    #       labels: list:len(bz){(post_NMS_boxes_per_image)}  (top category of each regressed box)
    #       image_size: tuple:len(2)
    # Output:
    #       projected_masks: list:len(bz){(post_NMS_boxes_per_image,image_size[0],image_size[1]
    def postprocess_mask(self, masks_outputs, boxes, labels, image_size=(800,1088)):
      
      
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      bz = len(labels)
      list_mask = []                     #  len(bz){boxes_per_bz,C,28,28}
      k = 0

      for i in range(bz):
        masks_bz = torch.zeros(len(labels[i]),image_size[0],image_size[1]).to(device)
        masks_bz = masks_outputs[k:k+len(labels[i])]
        list_mask.append(masks_bz)
        k+=len(labels[i])

      projected_masks = []
      for i in range(bz):
        projected_masks_bz = self.postprocess_mask_single(list_mask[i],boxes[i],labels[i],image_size)
        projected_masks.append(projected_masks_bz)

      return projected_masks


    # This function processes the mask for a single image and projects it to a 800x1088 format

    def postprocess_mask_single(self,list_mask_bz,boxes_bz,labels_bz,image_size=(800,1088)):
      
      projected_masks_bz = torch.zeros(len(labels_bz),image_size[0],image_size[1])
      # print("lengths",len(labels_bz),len(boxes_bz))
      for i,box in enumerate(boxes_bz):
        # print(projected_masks_bz[i].shape)
        # print(list_mask_bz[i].shape)
        int_mask = list_mask_bz[i][labels_bz[i]-1]
        # print(projected_masks_bz[i].shape,int_mask.shape)
        box[0] = torch.clamp(box[0],0,800)
        box[1] = torch.clamp(box[1],0,1088)
        box[2] = torch.clamp(box[2],0,800)
        box[3] = torch.clamp(box[3],0,1088)
        try:                                            ##An error here means there are cross-boundary boxes which we will be skipping
          projected_masks_bz[i][int(box[1]):int(box[3]),int(box[0]):int(box[2])] = torch.nn.functional.interpolate(int_mask.unsqueeze(0).unsqueeze(0),size = (int(box[3])-int(box[1]),int(box[2])-int(box[0])),mode ='bilinear').squeeze(0).squeeze(0)
          continue
      return projected_masks_bz




    # Compute the total loss of the Mask Head
    # Input:
    #      mask_output: (total_boxes,C,2*P,2*P)
    #      labels: (total_boxes)
    #      gt_masks: (total_boxes,2*P,2*P)
    # Output:
    #      mask_loss
   def compute_loss(self,mask_output,labels,gt_masks):

      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


      pred_masks = torch.zeros(mask_output.shape[0],28,28)
      for i in range(len(mask_output)):
        pred_masks[i] = mask_output[i][labels[i]-1]

      pred = pred_masks.view(-1).to(device)
      gt = gt_masks.view(-1).to(device)
      criterion = torch.nn.BCELoss()
      mask_loss = criterion(pred,gt) 

      return mask_loss



    # Forward the pooled feature map Mask Head
    # Input:
    #        features: (total_boxes, 256,P,P)
    # Outputs:
    #        mask_outputs: (total_boxes,C,2*P,2*P)
    def forward(self, features):

      mask_outputs = self.MaskBranch(features)

      return mask_outputs

if __name__ == '__main__':

	def get_projected_mask(mask,box):
	  projected_mask = torch.zeros(800,1088)
	  # print(mask.shape)
	  # print(mask.unsqueeze(0).unsqueeze(0).shape)
	  box[0] = torch.clamp(box[0],0,800)
	  box[1] = torch.clamp(box[1],0,1088)
	  box[2] = torch.clamp(box[2],0,800)
	  box[3] = torch.clamp(box[3],0,1088)
	  try:
	    projected_mask[int(box[0]):int(box[2]),int(box[1]):int(box[3])] = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0),size = (int(box[2])-int(box[0]),int(box[3])-int(box[1])),mode ='bilinear',align_corners=False).squeeze(0).squeeze(0)
	  except:
	    return projected_mask
	  return projected_mask


	def plot(backbone, rpn, box_head_net, mask_head_net, train_loader, test_loader, resume_checkpoint = None):
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
	  

	  for batch_idx, data in enumerate(test_loader):
	      # Get raw data
	      images=data['images'].to(device)
	      len_images = len(images)
	      indexes=data['index']
	      gt_labels=data['labels']
	      gt_boxes=[bbox.to(device) for bbox in data['bbox']]
	      masks=[mask.to(device) for mask in data['masks']]

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
	      try:
	        boxes, scores, labels, gt_masks, props = mask_head_net.preprocess_ground_truth_creation(
	            proposals,
	            class_logits,
	            box_preds,
	            gt_labels,
	            gt_boxes,
	            masks
	            )
	      except:
	        print("skipping")
	        continue
	      color_list = ["jet", "bone", "Spectral"]
	      for i in range(len(boxes)):
	        plot_img = transforms.functional.normalize(images[i,:,:,:].to('cpu'),(-0.485/0.229,-0.456/0.224,-0.406/0.225),(1/0.229,1/0.224,1/0.225)) 
	        plot_img = plot_img.permute(1,2,0)
	        fig,ax=plt.subplots(1,1)
	        ax.imshow(plot_img)
	        refined_masks = torch.zeros(3,800,1088).to(device)

	        for idx,gtMask in enumerate(gt_masks[i]):

	          projected_mask = get_projected_mask(gtMask,boxes[i][idx])
	          cate = labels[i][idx]
	          refined_masks[int(cate.item()) - 1] += projected_mask.to(device)


	        for idx,ref_mask in enumerate(refined_masks):

	          color = color_list[idx]
	          refined_masks[idx] = torch.clamp(refined_masks[idx],0,1)
	          mask_new = np.ma.masked_where(refined_masks[idx].cpu().data.numpy() == 0, refined_masks[idx].cpu().data.numpy())
	          plt.imshow(mask_new, cmap=color, alpha=0.7)

	        plt.show()


	print("Building the dataset...")
	backbone, rpn, box_head_net, mask_head_net, train_loader, test_loader = build_dataset()

	print("Plotting ground truth...")
	plot(backbone, rpn, box_head_net, mask_head_net, train_loader, test_loader)
