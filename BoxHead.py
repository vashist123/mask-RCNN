import torch
import torch.nn.functional as F
from torch import nn
from utils_new import *
from rpn_new import *
from backbone import *
import time
import os

#################
# Detect CoLab
#################
try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

class BoxHead(torch.nn.Module):
    def __init__(self,Classes=3,P=7):
        super(BoxHead,self).__init__()

        self.C=Classes
        self.P=P
        self.fc1 = nn.Linear(256*P*P,1024)
        self.fc2 = nn.Linear(1024,1024)
        self.classifier = nn.Linear(1024,4)
        self.regressor = nn.Linear(1024,12)    ## t_total= [t(1)x, t(1)y, t(1)w, t(1),h, . . . , t(C)x, t(C)y, t(C)w, t(C)h] format as given in pdf pg6



    #  This function assigns to each proposal either a ground truth box or the background class (we assume background class is 0)
    #  Input:
    #       proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #       gt_labels: list:len(bz) {(n_obj)}
    #       bboxes: list:len(bz){(n_obj, 4)}
    #  Output: (make sure the ordering of the proposals are consistent with MultiScaleRoiAlign)
    #       labels: (total_proposals,1) (the class that the proposal is assigned)
    #       regressor_target: (total_proposals,4) (target encoded in the [t_x,t_y,t_w,t_h] format)
    def create_ground_truth(self,proposals,gt_labels,bboxes,indexes):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Calculate shapes
        bz = len(proposals)
        per_image_proposals = proposals[0].shape[0]
        total_proposals = bz * per_image_proposals

        # Create output tensors
        labels = torch.zeros((total_proposals, 1), device = device)
        regressor_target = torch.zeros((total_proposals, 4), device = device)

        # Load the ground truth from cache dictionary
        if not hasattr(self, 'ground_dict'):
            self.ground_dict = {}
        key = str(indexes)
        if key in self.ground_dict:
            labels, regressor_target = self.ground_dict[key]
            return labels, regressor_target

        for img_idx in range(bz):
            for proposal_idx in range(per_image_proposals):
                # Get the current proposal
                proposal = proposals[img_idx][proposal_idx]
                proposal_width = (proposal[2] - proposal[0]).item()
                proposal_height = (proposal[3] - proposal[1]).item()
                proposal_center_x = (proposal[0] + proposal_width/2).item()
                proposal_center_y = (proposal[1] + proposal_height/2).item()
                cxcywh_proposal = [proposal_center_x, proposal_center_y, proposal_width, proposal_height]


                # Initialize variables for keeping track of the max IOU
                max_iou = None
                max_label = None
                max_target = None

                for obj_idx in range(len(bboxes[img_idx])):
                    # Get the current bbox
                    bbox = bboxes[img_idx][obj_idx]
                    bbox_width = (bbox[2] - bbox[0]).item()
                    bbox_height = (bbox[3] - bbox[1]).item()
                    bbox_center_x = (bbox[0] + bbox_width/2).item()
                    bbox_center_y = (bbox[1] + bbox_height/2).item()
                    cxcywh_bbox = [bbox_center_x, bbox_center_y, bbox_width, bbox_height]

                    # Compute IOU
                    iou = IOU(
                        bbox.cpu().data.numpy().flatten().tolist(),
                        proposal.cpu().data.numpy().flatten().tolist(),
                    )

                    # Keep track of the max IOU bbox for the proposal
                    if iou > .1:
                        if max_iou is None or iou > max_iou:
                            max_iou = iou
                            max_label = gt_labels[img_idx][obj_idx]
                            max_target = cxcywh_bbox

                if max_iou is not None:
                    flattened_idx = (img_idx * per_image_proposals) + proposal_idx
                    labels[flattened_idx] = max_label
                    transformed_bbox = encode_bbox(max_target, cxcywh_proposal)
                    regressor_target[flattened_idx][0] = transformed_bbox[0]
                    regressor_target[flattened_idx][1] = transformed_bbox[1]
                    regressor_target[flattened_idx][2] = transformed_bbox[2]
                    regressor_target[flattened_idx][3] = transformed_bbox[3]

        # Save the ground truth to cache dictionary
        self.ground_dict[key] = labels, regressor_target

        return labels, regressor_target



    # This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
    # a (256,P,P) feature map. This feature map is then flattened into a (256*P*P) vector
    # Input:
    #      fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
    #      proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #      P: scalar
    # Output:
    #      feature_vectors: (total_proposals, 256*P*P)  (make sure the ordering of the proposals are the same as the ground truth creation)
    def MultiScaleRoiAlign(self, fpn_feat_list,proposals,P=7):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print(fpn_feat_list.shape)
        #####################################
        # Here you can use torchvision.ops.RoIAlign check the docs
        #####################################
        bz = len(proposals)
        per_image_proposals = proposals[0].shape[0]
        total_proposals = bz * per_image_proposals
        stride = [4,8,16,32,32]
        # prop = torch.stack(proposals)
        # print(prop.shape)
        # propo = prop.reshape(-1,4)
        # print(proposal.shape)

        feature_vectors = torch.zeros(total_proposals,256*P*P).to(device)

        for img_idx in range(bz):
          for proposal_idx in range(per_image_proposals):

            proposal = proposals[img_idx][proposal_idx]
            width = (proposal[2] - proposal[0]).item()
            height = (proposal[3] - proposal[1]).item()
            k = 4 + np.log2(np.sqrt(width*height)/224)
            k = np.floor(min(max(2,k),5)).astype('int')

            stride_for_proposal = stride[k-2]
            propo = proposal/stride_for_proposal

            flattened_idx = (img_idx * per_image_proposals) + proposal_idx
            # bbox_for_pooling = proposal[flattened_idx]/stride_for_proposal
            a=[]
            b=propo.reshape(1,4)
            a.append(b)
            roi_aligned_feature = torchvision.ops.roi_align(torch.unsqueeze(fpn_feat_list[k-2][img_idx],dim=0),a,output_size = (P,P))
            roi_aligned_feature = torch.squeeze(roi_aligned_feature)
            roi_aligned_feature = torch.flatten(roi_aligned_feature)
            feature_vectors[flattened_idx] = roi_aligned_feature


        return feature_vectors



    # This function does the post processing for the results of the Box Head for a batch of images
    # Use the proposals to distinguish the outputs from each image
    # Input:
    #       class_logits: (total_proposals,(C+1))
    #       box_regression: (total_proposal,4*C)           ([t_x,t_y,t_w,t_h] format)
    #       proposals: list:len(bz)(per_image_proposals,4) (the proposals are produced from RPN [x1,y1,x2,y2] format)
    #       conf_thresh: scalar
    #       keep_num_preNMS: scalar (number of boxes to keep pre NMS)
    #       keep_num_postNMS: scalar (number of boxes to keep post NMS)
    # Output:
    #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)}  ([x1,y1,x2,y2] format)
    #       scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
    #       labels: list:len(bz){(post_NMS_boxes_per_image)}   (top class of each regressed box)
    def postprocess_detections(self, class_logits, box_regression, proposals, conf_thresh=0.5, keep_num_preNMS=50, keep_num_postNMS=9):
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      class_log = torch.nn.functional.softmax(class_logits)
      proposals = torch.stack(proposals)
      proposals = proposals.reshape(-1,4)
      
      scores,classes = torch.max(class_log,dim=1)
      zero_idx = torch.nonzero(classes[:]==0)
      scores[zero_idx[:]] = 0
      
      tresh_scores_ind = torch.nonzero(scores[:]>conf_thresh)
      new_scores = scores[tresh_scores_ind[:]].squeeze()
      new_classes = classes[tresh_scores_ind[:]].squeeze()
      new_proposals = proposals[tresh_scores_ind[:]].squeeze()
      new_box_regression = box_regression[tresh_scores_ind[:]].squeeze()
      boxes = torch.zeros(new_box_regression.shape[0],4).to(device)
      for idx,i in enumerate(new_box_regression):
        try:
          lab = (new_classes[idx]-1)*4
        except:
          return [], [], [], [], [], [] # An error here means no boxes have a high enough confidence
        boxes[idx][0:4] = i[lab:lab+4]
        
      x1y1x2y2_tresh = output_decodingd(boxes,new_proposals)
      try:
        new_sorted_scores,new_sorted_classes,sorted_x1y1x2y2 = zip(*[(x,y,z) for x,y,z in sorted(zip(new_scores,new_classes,x1y1x2y2_tresh), reverse = True )])
      except:
        print("BUG  - returning empty lists")
        return [], [], [], [], [], []
      
      preNMS_boxes = []
      preNMS_scores = []
      preNMS_labels = []
      found = 0
      for idx,i in enumerate(new_sorted_scores):
        if sorted_x1y1x2y2[idx][0] < 0 or sorted_x1y1x2y2[idx][1] < 0 or sorted_x1y1x2y2[idx][2] > 1088 or sorted_x1y1x2y2[idx][3] > 800:
          continue
        preNMS_scores.append(new_sorted_scores[idx])
        preNMS_boxes.append(sorted_x1y1x2y2[idx])
        preNMS_labels.append(new_sorted_classes[idx])
        found += 1
        if found >= keep_num_preNMS:
          break
      
      postNMS_scores,postNMS_boxes,postNMS_labels = self.NMSpost(preNMS_scores,preNMS_labels,preNMS_boxes,IOU_tresh = 0.5, keep_num_postNMS = keep_num_postNMS)
      
      return preNMS_boxes, preNMS_scores, preNMS_labels,postNMS_boxes,postNMS_scores,postNMS_labels


    def NMSpost(self,preNMS_scores,pre_NMS_labels,preNMS_boxes,IOU_tresh,keep_num_postNMS):
      postNMS_boxes = []
      posNMS_labels = []
      postNMS_scores = []
      
      next_NMS_boxes = []
      next_NMS_labels = []
      next_NMS_scores = []
      
      curr_NMS_boxes = preNMS_boxes
      curr_NMS_labels = pre_NMS_labels
      curr_NMS_scores = preNMS_scores
      c = [0,0,0,0]
      while(len(curr_NMS_scores)>0):
        x1 = curr_NMS_boxes[0][0]
        y1 = curr_NMS_boxes[0][1]
        x2 = curr_NMS_boxes[0][2]
        y2 = curr_NMS_boxes[0][3]
        lab = curr_NMS_labels[0]
        
        for idx,i in enumerate(curr_NMS_scores):
          if idx==0:
            c[curr_NMS_labels[idx]] += 1
            postNMS_scores.append(curr_NMS_scores[0].cpu().data.numpy())
            posNMS_labels.append(curr_NMS_labels[0].cpu().data.numpy())
            postNMS_boxes.append(curr_NMS_boxes[0].cpu().data.numpy())
            if len(postNMS_scores)>keep_num_postNMS:
              return postNMS_scores,postNMS_boxes,posNMS_labels
          else:
            iou = IOU([x1,y1,x2,y2],[curr_NMS_boxes[idx][0],curr_NMS_boxes[idx][1],curr_NMS_boxes[idx][2],curr_NMS_boxes[idx][3]])
            if iou > IOU_tresh and curr_NMS_labels[idx]==lab :
              continue
            elif curr_NMS_labels[idx]==1 and c[1]>2:
              continue
            elif curr_NMS_labels[idx]==2 and c[2]>2:
              continue
            elif curr_NMS_labels[idx]==3 and c[3]>2:
              continue
            else:
              next_NMS_scores.append(i)
              next_NMS_labels.append(curr_NMS_labels[idx])
              next_NMS_boxes.append(curr_NMS_boxes[idx])
        curr_NMS_boxes = next_NMS_boxes
        curr_NMS_scores = next_NMS_scores
        curr_NMS_labels = next_NMS_labels
        next_NMS_scores = []
        next_NMS_labels = []
        next_NMS_boxes = []
      return postNMS_scores,postNMS_boxes,posNMS_labels




    # Compute the total loss of the classifier and the regressor
    # Input:
    #      class_logits: (total_proposals,(C+1)) (as outputed from forward, not passed from softmax so we can use CrossEntropyLoss)
    #      box_preds: (total_proposals,4*C)      (as outputed from forward)
    #      labels: (total_proposals,1)
    #      regression_targets: (total_proposals,4)
    #      l: scalar (weighting of the two losses)
    #      effective_batch: scalar
    # Outpus:
    #      loss: scalar
    #      loss_class: scalar
    #      loss_regr: scalar
    def compute_loss(self,class_logits, box_preds, labels, regression_targets,l=1,effective_batch=30):

      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      p_mask = torch.nonzero((labels != 0).float(), as_tuple=True)
      n_mask = torch.nonzero((labels == 0).float(), as_tuple=True)
      M = effective_batch
      M_pos = round(M*0.75)

      p_mask_subsampled = (p_mask[0][torch.randperm(p_mask[0].shape[0])][:M_pos],)
      M_positive = len(p_mask_subsampled[0])
      M_negative = M - M_positive
      n_mask_subsampled = (n_mask[0][torch.randperm(n_mask[0].shape[0])][:M_negative],)
      # print(len(n_mask_subsampled[0]))

      pos_class_logits = class_logits[p_mask_subsampled]
      background_class_logits = class_logits[n_mask_subsampled]
      pos_labels = labels[p_mask_subsampled]
      # background_labels = labels[n_mask_subsampled]
      classification_loss = self.class_loss(pos_class_logits,background_class_logits,pos_labels)

      pos_regression_targets = regression_targets[p_mask_subsampled]
      pos_box_preds = box_preds[p_mask_subsampled]
      regression_loss = self.regr_loss(pos_regression_targets,pos_box_preds,pos_labels,effective_batch)
      regression_loss = l*regression_loss

      loss = classification_loss + regression_loss

      return loss, classification_loss, regression_loss

    def class_loss(self,pos_class_logits,background_class_logits,pos_labels):
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      criterion = torch.nn.CrossEntropyLoss()
      pos = pos_class_logits.shape[0]
      neg = background_class_logits.shape[0]
      tot = pos+neg
      if pos == 0:
        lss = (neg*criterion(background_class_logits,torch.zeros(background_class_logits.shape[0]).type(torch.LongTensor).to(device)))/tot
      else:
        lss = (pos*criterion(pos_class_logits,torch.flatten(pos_labels).type(torch.LongTensor).to(device)) + neg*criterion(background_class_logits,torch.zeros(background_class_logits.shape[0]).type(torch.LongTensor).to(device)))/tot
      return lss

    def regr_loss(self,pos_regression_targets,pos_box_preds,pos_labels,effective_batch):
      
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      spec_box_pred = torch.zeros(pos_labels.shape[0],4).to(device)
      criterion = torch.nn.SmoothL1Loss(reduction='sum')
      reg_loss=0
      
      for idx,i in enumerate(pos_box_preds,0):
        label = ((pos_labels[idx]-1)*4).type(torch.IntTensor)
        spec_box_pred[idx] = i[label:label+4]
      fl_spec_box_pred = torch.flatten(spec_box_pred)
      fl_target_reg = torch.flatten(pos_regression_targets)

      
      reg_loss = criterion(fl_spec_box_pred, fl_target_reg)


      return reg_loss/effective_batch



    # Forward the pooled feature vectors through the intermediate layer and the classifier, regressor of the box head
    # Input:
    #        feature_vectors: (total_proposals, 256*P*P)
    # Outputs:
    #        class_logits: (total_proposals,(C+1)) (we assume classes are C classes plus background, notice if you want to use
    #                                               CrossEntropyLoss you should not pass the output through softmax here)
    #        box_pred:     (total_proposals,4*C)
    def forward(self, feature_vectors):
        x = self.fc1(feature_vectors)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        class_logits = self.classifier(x)
        box_pred = self.regressor(x)


        return class_logits, box_pred

if __name__ == '__main__':

    # Put the path were you save the given pretrained model
    if IN_COLAB:
        pretrained_path=os.path.join(HOMEWORK_FOLDER, "checkpoints/checkpoint680.pth")
        path = os.path.join(HOMEWORK_FOLDER, 'checkpoints')
    else:
        pretrained_path='checkpoints/checkpoint680.pth'
        path = os.path.join('.', 'checkpoints')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    backbone = Resnet50Backbone()
    rpn_net = RPNHead()

    # Load rpn_net checkpoint
    checkpoint = torch.load(
        os.path.join(path, 'rpn_epoch_19'),
        map_location=device
    )
    rpn_net=rpn_net.to(device)
    rpn_net.load_state_dict(checkpoint['model_state_dict'])
    rpn_net.eval()

    # we will need the ImageList from torchvision
    from torchvision.models.detection.image_list import ImageList

    if IN_COLAB:
        imgs_path = os.path.join(HOMEWORK_FOLDER, "data/hw3_mycocodata_img_comp_zlib.h5")
        masks_path = os.path.join(HOMEWORK_FOLDER, "data/hw3_mycocodata_mask_comp_zlib.h5")
        labels_path = os.path.join(HOMEWORK_FOLDER, "data/hw3_mycocodata_labels_comp_zlib.npy")
        bboxes_path = os.path.join(HOMEWORK_FOLDER, "data/hw3_mycocodata_bboxes_comp_zlib.npy")
    else:
        imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
        masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
        labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
        bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'

    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    # Standard Dataloaders Initialization
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size

    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    batch_size = 1
    print("batch size:", batch_size)
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()


    with torch.no_grad():
        for iter, data in enumerate(test_loader, 0):
            images=data['images'].to(device)
            len_images = len(images)
            indexes=data['index']
            labels=data['labels']
            boxes=[bbox.to(device) for bbox in data['bbox']]

            # Take the features from the backbone
            backout = backbone(images)
            X = [backout["0"], backout["1"], backout["2"], backout["3"], backout["pool"]]
            clas_out,regr_out = rpn_net.forward(X)

            # Get proposals from RPN
            keep_topK = 200
            _, _, _, proposals  = rpn_net.postprocess( clas_out, regr_out, IOU_thresh=0.5, keep_num_preNMS=keep_topK, keep_num_postNMS=keep_topK )
            del _

            #The final output is
            # A list of proposal tensors: list:len(bz){(keep_topK,4)}
            proposals = torch.stack([torch.stack(proposal, 0) for proposal in proposals], dim=0)
            proposals=[crop_to_image_size(proposal) for proposal in proposals]


            # A list of features produces by the backbone's FPN levels: list:len(FPN){(bz,256,H_feat,W_feat)}
            fpn_feat_list= list(backout.values())

            boxhead = BoxHead()
            start = time.time()
            labels,regressor_target = boxhead.create_ground_truth(proposals,labels,boxes,indexes)

            # Flatten the proposals and convert them to cx, xy, w, h format
            flattened_proposals = torch.stack(proposals, dim=0).reshape((regressor_target.shape))
            flattened_proposals_width = (flattened_proposals[:, 2] - flattened_proposals[:, 0])
            flattened_proposals_height = (flattened_proposals[:, 3] - flattened_proposals[:, 1])
            flattened_proposals_center_x = (flattened_proposals[:, 0] + flattened_proposals_width/2)
            flattened_proposals_center_y = (flattened_proposals[:, 1] + flattened_proposals_height/2)
            flattened_proposals[:, 0] = flattened_proposals_center_x
            flattened_proposals[:, 1] = flattened_proposals_center_y
            flattened_proposals[:, 2] = flattened_proposals_width
            flattened_proposals[:, 3] = flattened_proposals_height

            # Decode the output of the network
            decoded_coord=output_decoding(regressor_target, flattened_proposals)

            # Visualization of the proposals
            for i in range(batch_size):
                plot_img = transforms.functional.normalize(images[i,:,:,:].to('cpu'),(-0.485/0.229,-0.456/0.224,-0.406/0.225),(1/0.229,1/0.224,1/0.225))
                plot_img = plot_img.permute(1,2,0)
                fig,ax=plt.subplots(1,1)
                ax.imshow(plot_img)
                for proposal_idx, box in enumerate(proposals[i]):
                    label = labels[keep_topK * i + proposal_idx]
                    if label.item() == 0:
                        continue
                    box=box.view(-1)
                    gt_box=decoded_coord[keep_topK * i + proposal_idx].cpu().data.numpy().flatten().tolist()
                    rect=patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,color='b')
                    ax.add_patch(rect)
                    color_list = ["m", "r", "g"]
                    color = color_list[int(label.item()) - 1]
                    rect=patches.Rectangle((gt_box[0],gt_box[1]),gt_box[2]-gt_box[0],gt_box[3]-gt_box[1],fill=False,color=color)
                    ax.add_patch(rect)

                plt.show()
