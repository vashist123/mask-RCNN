import numpy as np
import torch
from functools import partial
import torchvision
import math

def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
  
    return tuple(map(list, zip(*map_results)))

# This function compute the IOU between two set of boxes 
def IOU(boxA, boxB):
    x1, y1, x2, y2 = boxA
    x1_hat, y1_hat, x2_hat, y2_hat = boxB

    # Intersection
    int_x1 = max(x1, x1_hat)
    int_y1 = max(y1, y1_hat)
    int_x2 = min(x2, x2_hat)
    int_y2 = min(y2, y2_hat)

    # Intersection Area
    intersection_area = (int_x2 - int_x1) * (int_y2 - int_y1)

    # Individual Areas
    area_hat = (x2_hat - x1_hat) * (y2_hat - y1_hat)
    area = (x2 - x1) * (y2 - y1)

    # Union Area
    union_area = (area + area_hat) - intersection_area

    # Add the IOU to the results
    try:
      iou = min(max(intersection_area / union_area, 0), 1)

      # Detect if there actually is no intersection between the bounding boxes, if so set IOU = 0
      if (x1_hat > x2 or x2_hat < x1) or (y1_hat > y2 or y2_hat < y1):
        iou = 0

      return iou
    except:
      # Division By Zero Error happens if the union is 0, in this case,
      # we set IOU = 0.
      return 0


def IOU_vec(x1, y1, x2, y2, x1_hat, y1_hat, x2_hat, y2_hat):
    # Intersection
    int_x1 = torch.max(x1, x1_hat)
    int_y1 = torch.max(y1, y1_hat)
    int_x2 = torch.min(x2, x2_hat)
    int_y2 = torch.min(y2, y2_hat)

    # Intersection Area
    intersection_area = (int_x2 - int_x1) * (int_y2 - int_y1)
    del int_x1, int_y1, int_x2, int_y2

    # Individual Areas
    area_hat = (x2_hat - x1_hat) * (y2_hat - y1_hat)
    area = (x2 - x1) * (y2 - y1)

    # Union Area
    union_area = (area + area_hat) - intersection_area
    del area, area_hat

    # Add the IOU to the results
    iou = torch.clamp(intersection_area / union_area, 0, 1) # Calculate IOU and make sure its between 0-1
    del intersection_area, union_area
    iou[iou!=iou] = 0 # Set division by zero NaNs to 0
    
    # Detect if there actually is no intersection between the bounding boxes, if so set IOU = 0
    iou[x1_hat > x2] = 0
    iou[x2_hat < x1] = 0
    iou[y1_hat > y2] = 0
    iou[y2_hat < y1] = 0

    return iou


# This function flattens the output of the network and the corresponding anchors
# in the sense that it concatenate  the outputs and the anchors from all the grid cells from all
# the FPN levels from all the images into 2D matrices
# Each row correspond of the 2D matrices corresponds to a specific grid cell
# Input:
#       out_r_list: list:len(FPN){(bz,num_anchors*4,grid_size[0],grid_size[1])}
#       out_c_list: list:len(FPN){(bz,num_anchors*1,grid_size[0],grid_size[1])}
#       anchors_list: list:len(FPN){(num_anchors*grid_size[0]*grid_size[1],4)}
# Output:
#       flatten_regr: (total_number_of_anchors*bz,4)
#       flatten_clas: (total_number_of_anchors*bz)
#       flatten_anchors: (total_number_of_anchors*bz,4)
def output_flattening(out_r_list, out_c_list, anchors_list):
    flatten_regr_list = []
    flatten_clas_list = []
    flatten_anchors_list = []

    for out_r, out_c, anchors in zip(out_r_list, out_c_list, anchors_list):
        # Get shapes
        bz = out_c.shape[0]
        num_anchors = out_c.shape[1]
        grid_size_0  = out_c.shape[2]
        grid_size_1  = out_c.shape[3]

        flatten_regr = out_r.permute(0,2,3,1).reshape((bz*grid_size_0*grid_size_1*num_anchors, 4))
        flatten_clas = torch.flatten(out_c.permute(0,2,3,1))
        flatten_anchors = anchors.unsqueeze(0).expand(bz, -1, -1).reshape((bz*grid_size_0*grid_size_1*num_anchors, 4))

        flatten_regr_list.append(flatten_regr)
        flatten_clas_list.append(flatten_clas)
        flatten_anchors_list.append(flatten_anchors)

    return torch.cat(flatten_regr_list, 0), torch.cat(flatten_clas_list, 0), torch.cat(flatten_anchors_list, 0)

def encode_bbox(bbox, anchor_bbox):
    x, y, w, h = bbox
    x_a, y_a, w_a, h_a = anchor_bbox
    # Encodes using the calculation given in the PDF handout on Page 6
    return (x-x_a)/w_a, (y-y_a)/h_a, math.log(w/w_a), math.log(h/h_a) 

def encode_bbox_vec(bbox, anchor_bbox):
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    x_a, y_a, w_a, h_a = anchor_bbox[:, 0], anchor_bbox[:, 1], anchor_bbox[:, 2], anchor_bbox[:, 3]
    # Encodes using the calculation given in the PDF handout on Page 6
    return (x-x_a)/w_a, (y-y_a)/h_a, torch.log(w/w_a), torch.log(h/h_a) 

# This function decodes the output that are given in the [t_x,t_y,t_w,t_h] format
# into box coordinates where it returns the upper left and lower right corner of the bbox
# Input:
#       flatten_out: (total_number_of_anchors*bz,4)
#       flatten_anchors: (total_number_of_anchors*bz,4)
# Output:
#       box: (total_number_of_anchors*bz,4)
def output_decoding(flatten_out, flatten_anchors, device='cpu'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    box = torch.zeros(flatten_out.shape, device=device)

    # Inverses what encode_bbox does above ^^^^^^
    box[:, 0:2] = (flatten_out[:, 0:2] * flatten_anchors[:, 2:4]) + flatten_anchors[:, 0:2] # (out * w_a/h_a) + x_a/y_a
    box[:, 2:4] = torch.exp(flatten_out[:, 2:4]) * flatten_anchors[:, 2:4] # e ^ (out) * w_a/h_a

    # Convert to x1, y1, x2, y2 format instead of cx, cy, w, h format
    converted_box = torch.zeros(flatten_out.shape, device=device)
    converted_box[:, 0] = box[:, 0] - box[:, 2]/2
    converted_box[:, 1] = box[:, 1] - box[:, 3]/2
    converted_box[:, 2] = box[:, 0] + box[:, 2]/2
    converted_box[:, 3] = box[:, 1] + box[:, 3]/2

    return converted_box

# This function decodes the output of the box head that are given in the [t_x,t_y,t_w,t_h] format
# into box coordinates where it return the upper left and lower right corner of the bbox
# Input:
#       regressed_boxes_t: (total_proposals,4) ([t_x,t_y,t_w,t_h] format)
#       flatten_proposals: (total_proposals,4) ([x1,y1,x2,y2] format)
# Output:
#       box: (total_proposals,4) ([x1,y1,x2,y2] format)
def output_decodingd(regressed_boxes_t,flatten_proposals, device='cpu'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    box = torch.zeros(regressed_boxes_t.shape[0],4).to(device)

    Wp = flatten_proposals[:,2]-flatten_proposals[:,0]
    Hp = flatten_proposals[:,3]-flatten_proposals[:,1]

    x_star = regressed_boxes_t[:,0]*Wp[:] + (flatten_proposals[:,2]+flatten_proposals[:,0])/2
    y_star = regressed_boxes_t[:,1]*Hp[:] + (flatten_proposals[:,3]+flatten_proposals[:,1])/2

    w_star = torch.exp(regressed_boxes_t[:,2])*Wp[:]
    h_star = torch.exp(regressed_boxes_t[:,3])*Hp[:]

    box[:,0] = x_star-(w_star/2)
    box[:,1] = y_star-(h_star/2)
    box[:,2] = x_star+(w_star/2)
    box[:,3] = y_star+(h_star/2)
    
    return box

# This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
# a (256,P,P) feature map. This feature map is then flattened into a (256*P*P) vector
# Input:
#      fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
#      proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
#      P: scalar
# Output:
#      feature_vectors: (total_proposals, 256*P*P)  (make sure the ordering of the proposals are the same as the ground truth creation)
def MultiScaleRoiAlign(fpn_feat_list,proposals,P=7):
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

def crop_to_image_size(proposals):
    proposals[:, 0] = torch.clamp(proposals[:, 0], 0, 1088)
    proposals[:, 1] = torch.clamp(proposals[:, 1], 0, 800)
    proposals[:, 2] = torch.clamp(proposals[:, 2], 0, 1088)
    proposals[:, 3] = torch.clamp(proposals[:, 3], 0, 800)
    return proposals
