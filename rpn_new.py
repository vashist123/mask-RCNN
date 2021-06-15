import torch
from torchvision import transforms
from torch.nn import functional as F
from torch import nn, Tensor
from dataset import *
from utils_new import *
import torchvision


class RPNHead(torch.nn.Module):
    # The input of the initialization of the RPN is:
    # Input:
    #       computed_anchors: the anchors computed in the dataset
    #       num_anchors: the number of anchors that are assigned to each grid cell
    #       in_channels: number of channels of the feature maps that are outputed from the backbone
    #       device: the device that we will run the model
    def __init__(self, num_anchors=3, in_channels=256, device='cuda',
                 anchors_param=dict(ratio=[[1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2]],
                                    scale=[32, 64, 128, 256, 512],
                                    grid_size=[(200, 272), (100, 136), (50, 68), (25, 34), (13, 17)],
                                    stride=[4, 8, 16, 32, 64])
                 ):
        super(RPNHead,self).__init__()
        self.device=device
        self.num_anchors = num_anchors
        self.in_channels = in_channels 
        self.anchors_param = anchors_param

        # TODO  Define Intermediate Layer
        self.intermediate = torch.nn.Sequential(
            torch.nn.Conv2d(256,256,kernel_size=3,padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
            )

        # TODO  Define Proposal Classifier Head
        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(256, num_anchors * 1,kernel_size=1,padding=0),
            torch.nn.Sigmoid()
            )

        # TODO Define Proposal Regressor Head
        self.regressor = torch.nn.Sequential(
            torch.nn.Conv2d(256, num_anchors * 4,kernel_size=1,padding=0)
            )

        #  find anchors
        self.anchors=self.create_anchors(self.anchors_param['ratio'],self.anchors_param['scale'],self.anchors_param['grid_size'],self.anchors_param['stride'])

    # Forward each level of the FPN output through the intermediate layer and the RPN heads
    # Input:
    #       X: list:len(FPN){(bz,256,grid_size[0],grid_size[1])}
    # Ouput:
    #       logits: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
    #       bbox_regs: list:len(FPN){(bz,4*num_anchors, grid_size[0],grid_size[1])}
    def forward(self, X):

        # forward_single over each FPN level
        logits, bbox_regs  = MultiApply(
            self.forward_single,
            X
        )

        return logits, bbox_regs

    # Forward a single level of the FPN output through the intermediate layer and the RPN heads
    # Input:
    #       feature: (bz,256,grid_size[0],grid_size[1])}
    # Ouput:
    #       logit: (bz,1*num_acnhors,grid_size[0],grid_size[1])
    #       bbox_regs: (bz,4*num_anchors, grid_size[0],grid_size[1])
    def forward_single(self, feature):

        #TODO forward through the Intermediate layer
        X = self.intermediate(feature)


        #TODO forward through the Classifier Head
        logit = self.classifier(X)


        #TODO forward through the Regressor Head
        bbox_reg = self.regressor(X)

        return logit, bbox_reg


    # This function creates the anchor boxes for all FPN level
    # Input:
    #       aspect_ratio: list:len(FPN){list:len(number_of_aspect_ratios)}
    #       scale:        list:len(FPN)
    #       grid_size:    list:len(FPN){tuple:len(2)}
    #       stride:        list:len(FPN)
    # Output:
    #       anchors_list: list:len(FPN){(grid_size[0]*grid_size[1]*num_anchors,4)}
    def create_anchors(self, aspect_ratio, scale, grid_size, stride):

        # create_anchors_single over each FPN level
        anchors_list = []
        for ar, sc, gs, st in zip(aspect_ratio, scale, grid_size, stride):
            anchors_list.append(self.create_anchors_single(ar, sc, gs, st))

        return anchors_list



    # This function creates the anchor boxes for one FPN level
    # Input:
    #      aspect_ratio: list:len(number_of_aspect_ratios)
    #      scale: scalar
    #      grid_size: tuple:len(2)
    #      stride: scalar
    # Output:
    #       anchors: (grid_size[0]*grid_size[1]*num_acnhors,4)
    def create_anchors_single(self, aspect_ratio, scale, grid_size, stride):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Create anchors tensor
        num_anchors = len(aspect_ratio)
        anchors = torch.zeros((grid_size[0], grid_size[1], num_anchors, 4))

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                for k in range(num_anchors):
                    # Calculate height and width of anchor
                    h = ((scale ** 2) / aspect_ratio[k]) ** .5
                    w = (aspect_ratio[k] * h)

                    center_h = (i * stride) + (stride / 2)
                    center_w = (j * stride) + (stride / 2)
                    
                    # Done the same way as: https://piazza.com/class/kedjeyrln6icm?cid=292_f2
                    # The order in the Piazza post is x, y, w, h
                    anchors[i][j][k][0] = center_w
                    anchors[i][j][k][1] = center_h
                    anchors[i][j][k][2] = w
                    anchors[i][j][k][3] = h

        anchors = anchors.reshape((grid_size[0]*grid_size[1]*num_anchors, 4)).to(device)

        assert anchors.shape == (grid_size[0]*grid_size[1]*num_anchors,4)
        return anchors

    def get_anchors(self):
        return self.anchors

    # This function creates the ground truth for a batch of images
    # Input:
    #      bboxes_list: list:len(bz){(number_of_boxes,4)}
    #      indexes: list:len(bz)
    #      image_shape: list:len(bz){tuple:len(2)}
    # Ouput:
    #      ground: list:len(FPN){(bz,num_anchors,grid_size[0],grid_size[1])}
    #      ground_coord: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
    def create_batch_truth(self, bboxes_list, indexes, image_shape):
        bz = len(bboxes_list)
        num_fpn_levels = len(self.anchors_param['scale'])

        ground = [[] for fpn_idx in range(num_fpn_levels)]
        ground_coord = [[] for fpn_idx in range(num_fpn_levels)]

        # Call create_ground_truth on each image in the batch
        for idx in range(bz):
            ground_clas_img, ground_coord_img  = self.create_ground_truth(bboxes_list[idx], indexes[idx], self.anchors_param['grid_size'], self.get_anchors(), image_shape[idx])
            for fpn_idx in range(num_fpn_levels):
                ground[fpn_idx].append(ground_clas_img[fpn_idx].unsqueeze(0))
                ground_coord[fpn_idx].append(ground_coord_img[fpn_idx].unsqueeze(0))

        # Convert the list of len(bz) into a torch tensor
        for fpn_idx in range(num_fpn_levels):
            ground[fpn_idx] = torch.cat(ground[fpn_idx], 0)
            ground_coord[fpn_idx] = torch.cat(ground_coord[fpn_idx], 0)

        return ground, ground_coord

    # This function create the ground truth for one image for all the FPN levels
    # It also caches the ground truth for the image using its index
    # Input:
    #       bboxes:      (n_boxes,4)
    #       index:       scalar (the index of the image in the total dataset)
    #       grid_size:   list:len(FPN){tuple:len(2)}
    #       anchor_list: list:len(FPN){(num_anchors*grid_size[0]*grid_size[1],4)}
    #       image_size:  {tuple:len(2)}
    # Output:
    #       ground_clas: list:len(FPN){(num_anchors,grid_size[0],grid_size[1])}
    #       ground_coord: list:len(FPN){(4*num_anchors,grid_size[0],grid_size[1])}
    def create_ground_truth(self, bboxes, index, grid_sizes, anchors_list, image_size):
        if not hasattr(self, 'ground_dict'):
            self.ground_dict = {}
        key = str(index)
        if key in self.ground_dict:
            groundt, ground_coord = self.ground_dict[key]
            return groundt, ground_coord

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        num_fpn_levels = len(grid_sizes)
        num_anchors = len(self.anchors_param['ratio'][0])

        ground_clas = []
        ground_coord = []

        max_iou = [-1 for _ in range(len(bboxes))]
        max_iou_fpn_idxs = [None for _ in range(len(bboxes))]
        max_iou_other_idxs = [None for _ in range(len(bboxes))]
        max_anchor_cxcywh_bboxes = [None for _ in range(len(bboxes))]

        for fpn_idx in range(num_fpn_levels):
            grid_size = grid_sizes[fpn_idx]
            anchors = anchors_list[fpn_idx].reshape((grid_size[0], grid_size[1], num_anchors, 4))

            # This function was completed using this Piazza post as reference:
            # https://piazza.com/class/kedjeyrln6icm?cid=296
            ground_clas_fpn = torch.ones((grid_size[0], grid_size[1], num_anchors), device = device)
            ground_clas_fpn = ground_clas_fpn * -1 # Fill class with negative ones inititally as neither positive (1) nor negative (0)
            ground_coord_fpn = torch.zeros((grid_size[0], grid_size[1], num_anchors, 4), device = device) # we will permute these dimensions later to the correct shape
            max_iou_of_anchor = torch.zeros((grid_size[0], grid_size[1], num_anchors), device = device)

            # For each bbox
            for bbox_idx, bbox in enumerate(bboxes):
                bbox_width = (bbox[2] - bbox[0])
                bbox_height = (bbox[3] - bbox[1])
                bbox_center_x = (bbox[0] + bbox_width/2)
                bbox_center_y = (bbox[1] + bbox_height/2)
                cxcywh_bbox = torch.clone(bbox)
                cxcywh_bbox[0] = bbox_center_x
                cxcywh_bbox[1] = bbox_center_y
                cxcywh_bbox[2] = bbox_width
                cxcywh_bbox[3] = bbox_height


                # For each anchor box, assign anchor boxes to this bbox if IOU > .7
                anchor_x1 = anchors[:, :, :, 0] - anchors[:, :, :, 2]/2
                anchor_y1 = anchors[:, :, :, 1] - anchors[:, :, :, 3]/2
                anchor_x2 = anchors[:, :, :, 0] + anchors[:, :, :, 2]/2
                anchor_y2 = anchors[:, :, :, 1] + anchors[:, :, :, 3]/2
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                ious = IOU_vec(anchor_x1, anchor_y1, anchor_x2, anchor_y2, bbox[0], bbox[1], bbox[2], bbox[3])
                
                # Ignore where the anchor box crosses the boundaries of the image
                ious[anchor_x1 < 0] = -1
                ious[anchor_y1 < 0] = -1
                ious[anchor_x2 > image_size[1]] = -1
                ious[anchor_y2 > image_size[0]] = -1
                
                # Keep track of the maximum IOU of each anchor box
                max_iou_of_anchor = torch.max(max_iou_of_anchor, ious)

                # Keep track of the maximum IOU of the current bbox
                cur_max_iou = torch.max(ious).item()
                if cur_max_iou > max_iou[bbox_idx] * 1.01:
                    max_iou[bbox_idx] = cur_max_iou
                    max_iou_fpn_idxs[bbox_idx] = fpn_idx
                    max_iou_other_idxs[bbox_idx] = torch.nonzero(torch.logical_or((ious == max_iou[bbox_idx]), ious > (.99 * max_iou[bbox_idx])), as_tuple = True)
                    max_anchor_cxcywh_bboxes[bbox_idx] = anchors[max_iou_other_idxs[bbox_idx]]

                # Assign the anchor box to this bbox (positive) where the IOU is high
                where_iou_is_high = torch.nonzero(ious > .7, as_tuple=True)
                ground_clas_fpn[where_iou_is_high] = 1
                transformed_bbox = encode_bbox_vec(cxcywh_bbox, anchors[where_iou_is_high])
                ground_coord_fpn[where_iou_is_high][:, 0] = transformed_bbox[0]
                ground_coord_fpn[where_iou_is_high][:, 1] = transformed_bbox[1]
                ground_coord_fpn[where_iou_is_high][:, 2] = transformed_bbox[2]
                ground_coord_fpn[where_iou_is_high][:, 3] = transformed_bbox[3]

            # For each anchor box, mark the anchor box as negative if max IOU to every bbox < .3
            # and the anchor box was not already marked as positive
            anchor_x1 = anchors[:, :, :, 0] - anchors[:, :, :, 2]/2
            anchor_y1 = anchors[:, :, :, 1] - anchors[:, :, :, 3]/2
            anchor_x2 = anchors[:, :, :, 0] + anchors[:, :, :, 2]/2
            anchor_y2 = anchors[:, :, :, 1] + anchors[:, :, :, 3]/2
            # Ignore where the anchor box crosses the boundaries of the image
            crosses_boundaries = torch.logical_or(torch.logical_or(torch.logical_or(anchor_x1 < 0, anchor_y1 < 0), anchor_x2 > image_size[1]), anchor_y2 > image_size[0])
            del anchor_x1, anchor_y1, anchor_x2, anchor_y2
            not_crosses_boundaries = torch.logical_not(crosses_boundaries)
            where_max_iou_is_low = torch.nonzero(torch.logical_and(torch.logical_and(max_iou_of_anchor < .3, ground_clas_fpn != 1), not_crosses_boundaries), as_tuple = True)
            ground_clas_fpn[where_max_iou_is_low] = 0
            del crosses_boundaries, not_crosses_boundaries, where_max_iou_is_low

            # Add to final output
            ground_clas.append(ground_clas_fpn.permute(2, 0, 1))
            ground_coord.append(ground_coord_fpn)


        # Also assign the anchor box (positive) to this bbox which returned the maximum IOU
        # for this bbox even if IOU < .7
        for bbox_idx, bbox in enumerate(bboxes):
            bbox_width = (bbox[2] - bbox[0])
            bbox_height = (bbox[3] - bbox[1])
            bbox_center_x = (bbox[0] + bbox_width/2)
            bbox_center_y = (bbox[1] + bbox_height/2)
            cxcywh_bbox = torch.clone(bbox)
            cxcywh_bbox[0] = bbox_center_x
            cxcywh_bbox[1] = bbox_center_y
            cxcywh_bbox[2] = bbox_width
            cxcywh_bbox[3] = bbox_height
            for max_iou_fpn_idx, max_iou_anchor_idx, max_iou_i, max_iou_j, max_anchor_cxcywh_bbox in zip([max_iou_fpn_idxs[bbox_idx]]*len(max_iou_other_idxs[bbox_idx]), max_iou_other_idxs[bbox_idx][2], max_iou_other_idxs[bbox_idx][0], max_iou_other_idxs[bbox_idx][1], max_anchor_cxcywh_bboxes[bbox_idx]):
                ground_clas[max_iou_fpn_idx][max_iou_anchor_idx, max_iou_i, max_iou_j] = 1
                transformed_bbox = encode_bbox(cxcywh_bbox, max_anchor_cxcywh_bbox.cpu().data.numpy())
                ground_coord[max_iou_fpn_idx][max_iou_i, max_iou_j, max_iou_anchor_idx, 0] = transformed_bbox[0]
                ground_coord[max_iou_fpn_idx][max_iou_i, max_iou_j, max_iou_anchor_idx, 1] = transformed_bbox[1]
                ground_coord[max_iou_fpn_idx][max_iou_i, max_iou_j, max_iou_anchor_idx, 2] = transformed_bbox[2]
                ground_coord[max_iou_fpn_idx][max_iou_i, max_iou_j, max_iou_anchor_idx, 3] = transformed_bbox[3]

        # Put ground_coord_fpn into the shape for returning
        for fpn_idx in range(num_fpn_levels):
            grid_size = grid_sizes[fpn_idx]
            ground_coord[fpn_idx] = ground_coord[fpn_idx].permute(2, 3, 0, 1).reshape((4*num_anchors,grid_size[0],grid_size[1]))

        self.ground_dict[key] = (ground_clas, ground_coord)

        return ground_clas, ground_coord

    # Compute the loss of the classifier
    # Input:
    #      p_out:     (positives_on_mini_batch)  (output of the classifier for sampled anchors with positive gt labels)
    #      n_out:     (negatives_on_mini_batch) (output of the classifier for sampled anchors with negative gt labels
    def loss_class(self, p_out, n_out):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loss = torch.nn.BCELoss()
        if len(p_out) == 0:
            return loss(n_out, torch.zeros(n_out.shape, device=device))
        else:
            return (loss(p_out, torch.ones(p_out.shape, device=device)) + loss(n_out, torch.zeros(n_out.shape, device=device))) / 2 # The BCELoss over the predicted/ground-truth positive and negative labels

    # Compute the loss of the regressor
    # Input:
    #       pos_target_coord: (positive_on_mini_batch,4) (ground truth of the regressor for sampled anchors with positive gt labels)
    #       pos_out_r: (positive_on_mini_batch,4)        (output of the regressor for sampled anchors with positive gt labels)
    def loss_reg(self, pos_target_coord, pos_out_r):
        loss = torch.nn.SmoothL1Loss()
        return loss(pos_out_r, pos_target_coord)

    # Compute the total loss for the FPN heads
    # Input:
    #       clas_out_list: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
    #       regr_out_list: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
    #       targ_clas_list: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
    #       targ_regr_list: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
    #       l: weighting lambda between the two losses
    # Output:
    #       loss: scalar
    #       loss_c: scalar
    #       loss_r: scalar
    def compute_loss(self, clas_out_list, regr_out_list, targ_clas_list, targ_regr_list, l=1, effective_batch=[50, 50, 50, 50, 50]):
        num_fpn_levels = len(clas_out_list)

        total_M, total_M_positive = 0, 0
        total_loss_c, total_loss_r  = None, None

        for fpn_idx, (clas_out, regr_out, targ_clas, targ_regr) in enumerate(zip(clas_out_list, regr_out_list, targ_clas_list, targ_regr_list)):
            # Completed in accordance with this Piazza post:
            # https://piazza.com/class/kedjeyrln6icm?cid=298

            # Get shapes
            bz = clas_out.shape[0]
            num_anchors = clas_out.shape[1]
            grid_size_0  = clas_out.shape[2]
            grid_size_1  = clas_out.shape[3]

            # Calculate masks for positive and negative labels
            p_mask = torch.nonzero(torch.flatten((targ_clas == 1).float()), as_tuple=True)
            n_mask = torch.nonzero(torch.flatten((targ_clas == 0).float()), as_tuple=True)

            # Subsample the positive and negative masks to get them to about an equal ratio
            M = effective_batch[fpn_idx]
            total_M += M
            M_half = round(M * .5)
            p_mask_subsampled = (p_mask[0][torch.randperm(p_mask[0].shape[0])][:M_half],) # Randomly shuffle the positive anchors and choose M/2 of them
            M_positive = len(p_mask_subsampled[0])
            total_M_positive += M_positive
            M_negative = M - M_positive # Fill in the remaining (M - M_positive) anchors with negative anchors
            n_mask_subsampled = (n_mask[0][torch.randperm(n_mask[0].shape[0])][:M_negative],) # Randomly shuffle the negative anchors and choose M_remaining (M_negative) of them
            p_mask = p_mask_subsampled
            n_mask = n_mask_subsampled

            # Perform loss_class
            p_out = torch.flatten(clas_out) # Flatten the class predictions
            n_out = torch.flatten(clas_out) # Flatten the class predictions
            p_targ = torch.flatten(targ_clas) # Flatten the class predictions
            n_targ = torch.flatten(targ_clas) # Flatten the class predictions
            p_out = p_out[p_mask] # Only get the predictions for the positive labels and flatten
            n_out = n_out[n_mask] # Only get the predicitions for the negative labels and flatten
            p_targ = p_targ[p_mask] # Only get the ground truth for the positive labels and flatten
            n_targ = n_targ[n_mask] # Only get the ground truth for the negative labels and flatten 
            loss_c = self.loss_class(p_out, n_out)

            # Perform loss_reg
            pos_out_r = regr_out.permute(0,2,3,1).reshape((bz*num_anchors*grid_size_0*grid_size_1, 4))  # Flatten the regression bbox predictions 
            pos_target_coord = targ_regr.permute(0,2,3,1).reshape((bz*num_anchors*grid_size_0*grid_size_1, 4))  # Flatten the regression bbox ground truth
            pos_out_r = pos_out_r[p_mask] # Only get the predictions for the regression bboxes for the positive labels
            pos_target_coord = pos_target_coord[p_mask] # Only get the ground truth for the regression bboxes for the positive labels
            if len(p_out) != 0:
                loss_r = self.loss_reg(pos_target_coord, pos_out_r)
            else:
                loss_r = 0

            # Equally weighing the loss_c and loss_r
            loss_c = l * loss_c # loss_c is over M anchors, weight with lambda

            if fpn_idx == 0:
                total_loss_c, total_loss_r = loss_c, loss_r
            else:
                total_loss_c += loss_c
                total_loss_r += loss_r

        total_loss_r = (total_M / total_M_positive) * total_loss_r # loss_r is only over M_positive anchors, so we weight it up accordingly
        total_loss = total_loss_c + total_loss_r

        return total_loss, total_loss_c, total_loss_r



    # Post process for the outputs for a batch of images
    # Input:
    #       out_c: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
    #       out_r: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
    #       IOU_thresh: scalar that is the IOU threshold for the NMS
    #       keep_num_preNMS: number of masks we will keep from each image before the NMS
    #       keep_num_postNMS: number of masks we will keep from each image after the NMS
    # Output:
    #       nms_clas_list: list:len(bz){(Post_NMS_boxes)} (the score of the boxes that the NMS kept)
    #       nms_prebox_list: list:len(bz){(Post_NMS_boxes,4)} (the coordinate of the boxes that the NMS kept)
    def postprocess(self, out_c, out_r, IOU_thresh=0.5, keep_num_preNMS=50, keep_num_postNMS=10):
        num_fpn_levels = len(out_c)
        bz = out_c[0].shape[0]

        # Convert out_c to (len(bz), len(FPN), 1*num_anchors,grid_size[0],grid_size[1])
        # Convert out_r to (len(bz), len(FPN), 4*num_anchors,grid_size[0],grid_size[1])
        reshaped_out_c = []
        reshaped_out_r = []
        for idx in range(bz):
            reshaped_out_c.append([])
            reshaped_out_r.append([])
            for fpn_idx in range(num_fpn_levels):
                reshaped_out_c[idx].append(out_c[fpn_idx][idx])
                reshaped_out_r[idx].append(out_r[fpn_idx][idx])

        # Now MultiApply over each image in the batch
        clas_list, prebox_list, nms_clas_list, nms_prebox_list = MultiApply(
            self.postprocessImg,
            reshaped_out_c,
            reshaped_out_r,
            IOU_thresh = IOU_thresh,
            keep_num_preNMS = keep_num_preNMS,
            keep_num_postNMS = keep_num_postNMS,
        )

        return clas_list, prebox_list, nms_clas_list, nms_prebox_list

    # Post process the output for one image
    # Input:
    #      mat_clas: list:len(FPN){(1,1*num_anchors,grid_size[0],grid_size[1])}  (score of the output boxes)
    #      mat_coord: list:len(FPN){(1,4*num_anchors,grid_size[0],grid_size[1])} (encoded coordinates of the output boxess)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4)
    def postprocessImg(self, mat_clas, mat_coord, IOU_thresh, keep_num_preNMS, keep_num_postNMS):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Add extra dimension
        mat_clas = [x.unsqueeze(0) for x in mat_clas]
        mat_coord = [x.unsqueeze(0) for x in mat_coord]

        # Get shape information
        grid_size_0  = mat_clas[0].shape[2]
        grid_size_1  = mat_clas[0].shape[3]

        # Flatten the inputs
        flatten_coord, flatten_clas, flatten_anchors = output_flattening(mat_coord, mat_clas, self.get_anchors())
        scores_list = torch.flatten(flatten_clas).cpu().data.numpy().flatten().tolist()
        decoded_coord = output_decoding(flatten_coord,flatten_anchors)

        # Initialize output
        clas = []
        prebox = []

        # Get the indexes of the highest scores to the lowest scores
        sorted_idxs = [i[0] for i in sorted(enumerate(scores_list), key=lambda x:x[1], reverse = True)]
        found = 0

        # Keep the top K highest scores
        for idx in sorted_idxs:
            x1y1x2y2_bbox = decoded_coord[idx]
            x1 = (x1y1x2y2_bbox[0]).item()
            y1 = (x1y1x2y2_bbox[1]).item()
            x2 = (x1y1x2y2_bbox[2]).item()
            y2 = (x1y1x2y2_bbox[3]).item()

            if x1 < 0 or y1 < 0 or x2 > 1088 or y2 > 800:
                # The bounding box crosses the boundaries of the image, we will ignore it
                continue

            clas.append(flatten_clas[idx])
            prebox.append(decoded_coord[idx])

            found += 1
            if found >= keep_num_preNMS:
                break

        # Perform NMS on the these boxes
        nms_clas, nms_prebox = self.NMS_vec(clas, prebox, thresh = IOU_thresh, keep_num_postNMS = keep_num_postNMS)
        nms_clas = nms_clas[:keep_num_postNMS]
        nms_prebox = nms_prebox[:keep_num_postNMS]

        return clas, prebox, nms_clas, nms_prebox # Returning both the pre-NMS and post-NMS results so we can plot them for the report

    # Input:
    #       clas: (top_k_boxes) (scores of the top k boxes)
    #       prebox: (top_k_boxes,4) (coordinate of the top k boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4)
    def NMS(self, clas, prebox, thresh, keep_num_postNMS):
        nms_clas = []
        nms_prebox = []

        cur_clas = [x for x in clas]
        cur_prebox = [x for x in prebox]
        next_clas = []
        next_prebox = []

        # NMS algorithm loop
        while len(cur_clas) > 0:
            cxcywh_bbox = cur_prebox[0]
            x1 = (cxcywh_bbox[0] - cxcywh_bbox[2]/2).item()
            y1 = (cxcywh_bbox[1] - cxcywh_bbox[3]/2).item()
            x2 = (cxcywh_bbox[0] + cxcywh_bbox[2]/2).item()
            y2 = (cxcywh_bbox[1] + cxcywh_bbox[3]/2).item()

            for i, clas in enumerate(cur_clas):
                if i == 0:
                    # Always keep the highest scoring prediction in any iteration of NMS
                    nms_clas.append(clas)
                    nms_prebox.append(cur_prebox[i])
                    if len(nms_clas) >= keep_num_postNMS:
                        return nms_clas,nms_prebox
                else:
                    cxcywh_bbox_i = cur_prebox[i]
                    x1_i = (cxcywh_bbox_i[0] - cxcywh_bbox_i[2]/2).item()
                    y1_i = (cxcywh_bbox_i[1] - cxcywh_bbox_i[3]/2).item()
                    x2_i = (cxcywh_bbox_i[0] + cxcywh_bbox_i[2]/2).item()
                    y2_i = (cxcywh_bbox_i[1] + cxcywh_bbox_i[3]/2).item()
                    iou = IOU(
                            [x1, y1, x2, y2],
                            [x1_i, y1_i, x2_i, y2_i],
                        )
                    if iou > thresh:
                        # Overlaps too much with a box we are already keeping, discard for the next loop of NMS
                        pass
                    else:
                        # Don't discard for the next loop of NMS
                        next_clas.append(clas)
                        next_prebox.append(cur_prebox[i])
            cur_clas = next_clas
            cur_prebox = next_prebox
            next_clas = []
            next_prebox = []


        return nms_clas,nms_prebox

    # Normal NMS is too slow, we need MatrixNMS
    # Original author: Francisco Massa:
    # https://github.com/fmassa/object-detection.torch
    # Ported to PyTorch by Max deGroot (02/01/2017)
    def NMS_vec(self, scores, boxes, thresh=0.5, keep_num_postNMS=200):
        """Apply non-maximum suppression at test time to avoid detecting too many
        overlapping bounding boxes for a given object.
        Args:
            scores: (tensor) The class predscores for the img, Shape:[num_priors].
            boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
            thresh: (float) The overlap thresh for suppressing unnecessary boxes.
            keep_num_postNMS: (int) The Maximum number of box preds to consider.
        Return:
           nms_clas: (Post_NMS_boxes)
           nms_prebox: (Post_NMS_boxes,4)
        """
        with torch.no_grad():
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if (type(scores) == list):
                scores = torch.stack(scores, 0).to(device)
            if (type(boxes) == list):
                boxes = torch.stack(boxes, 0).to(device)

            keep = scores.new(scores.size(0)).zero_().long()
            if boxes.numel() == 0:
                return keep
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]
            area = torch.mul(x2 - x1, y2 - y1)
            v, idx = scores.sort(0)  # sort in ascending order
            # I = I[v >= 0.01]
            idx = idx[-keep_num_postNMS:]  # indices of the top-k largest vals
            xx1 = boxes.new()
            yy1 = boxes.new()
            xx2 = boxes.new()
            yy2 = boxes.new()
            w = boxes.new()
            h = boxes.new()

            # keep = torch.Tensor()
            count = 0
            while idx.numel() > 0:
                i = idx[-1]  # index of current largest val
                # keep.append(i)
                keep[count] = i
                count += 1
                if idx.size(0) == 1:
                    break
                idx = idx[:-1]  # remove kept element from view
                # load bboxes of next highest vals
                torch.index_select(x1, 0, idx, out=xx1)
                torch.index_select(y1, 0, idx, out=yy1)
                torch.index_select(x2, 0, idx, out=xx2)
                torch.index_select(y2, 0, idx, out=yy2)
                # store element-wise max with next highest score
                xx1 = torch.clamp(xx1, min=x1[i])
                yy1 = torch.clamp(yy1, min=y1[i])
                xx2 = torch.clamp(xx2, max=x2[i])
                yy2 = torch.clamp(yy2, max=y2[i])
                w.resize_as_(xx2)
                h.resize_as_(yy2)
                w = xx2 - xx1
                h = yy2 - yy1
                # check sizes of xx1 and xx2.. after each iteration
                w = torch.clamp(w, min=0.0)
                h = torch.clamp(h, min=0.0)
                inter = w*h
                # IoU = i / (area(a) + area(b) - i)
                rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
                union = (rem_areas - inter) + area[i]
                IoU = inter.float()/union.float()  # store result in iou
                # keep only elements with an IoU <= thresh
                idx = idx[IoU.le(thresh)]
            
            nms_clas = [scores[idx] for idx in keep[0:keep_num_postNMS]]
            nms_prebox = [boxes[idx] for idx in keep[0:keep_num_postNMS]]
            return nms_clas, nms_prebox


if __name__ == "__main__":
    # file path and make a list

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

  
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    rpn_net = RPNHead()
    # push the randomized training data into the dataloader

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    torch.random.manual_seed(1)
    batch_size = 1
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()
    
    for i,batch in enumerate(train_loader,0):
        images=batch['images'][0,:,:,:]
        indexes=batch['index']
        boxes=batch['bbox']
        gt,ground_coord=rpn_net.create_batch_truth(boxes,indexes,[images.shape[-2:]]*len(images))


        # Flatten the ground truth and the anchors
        flatten_coord,flatten_gt,flatten_anchors=output_flattening(ground_coord,gt,rpn_net.get_anchors())
        
        # Decode the ground truth box to get the upper left and lower right corners of the ground truth boxes
        decoded_coord=output_decoding(flatten_coord,flatten_anchors)
        
        # Plot the image and the anchor boxes with the positive labels and their corresponding ground truth box
        plot_img = images
        plot_img = transforms.functional.normalize(plot_img,(-0.485/0.229,-0.456/0.224,-0.406/0.225),(1/0.229,1/0.224,1/0.225))
        plot_img = plot_img.permute(1,2,0)
        fig,ax=plt.subplots(1,1)
        ax.imshow(plot_img)
        
        find_cor=(flatten_gt==1).nonzero()
        find_neg=(flatten_gt==-1).nonzero()

        for elem in find_cor:
            coord=decoded_coord[elem,:].view(-1)
            anchor=flatten_anchors[elem,:].view(-1)

            col='r'
            rect=patches.Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],fill=False,color=col)
            ax.add_patch(rect)
            rect=patches.Rectangle((anchor[0]-anchor[2]/2,anchor[1]-anchor[3]/2),anchor[2],anchor[3],fill=False,color='b')
            ax.add_patch(rect)

        plt.show()
 
        if(i>40):
            break


