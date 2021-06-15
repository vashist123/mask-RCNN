from rpn import *
from BoxHead import *
import torch

if __name__ == '__main__':
    # ###
    # # Anchor Tests
    # ###
    # rpn_net = RPNHead()
    # truth = torch.load('test_cases/Anchor_Test/anchors_ratio0.5_scale128.pt')
    # output = rpn_net.create_anchors(.5 , 128, rpn_net.anchors_param['grid_size'], rpn_net.anchors_param['stride']).double()
    # print("Anchor Test 1", torch.all(torch.isclose(truth, output, .0001)))

    # truth = torch.load('test_cases/Anchor_Test/anchors_ratio0.8_scale512.pt')
    # output = rpn_net.create_anchors(.8, 512, rpn_net.anchors_param['grid_size'], rpn_net.anchors_param['stride']).double()
    # print("Anchor Test 2", torch.all(torch.isclose(truth, output, .0001)))

    # truth = torch.load('test_cases/Anchor_Test/anchors_ratio1_scale256.pt')
    # output = rpn_net.create_anchors(1, 256, rpn_net.anchors_param['grid_size'], rpn_net.anchors_param['stride']).double()
    # print("Anchor Test 3", torch.all(torch.isclose(truth, output, .0001)))

    # truth = torch.load('test_cases/Anchor_Test/anchors_ratio2_scale256.pt')
    # output = rpn_net.create_anchors(2, 256, rpn_net.anchors_param['grid_size'], rpn_net.anchors_param['stride']).double()
    # print("Anchor Test 4", torch.all(torch.isclose(truth, output, .0001)))

    # ###
    # # Ground Truth Tests
    # ###
    # rpn_net = RPNHead()
    # truth = torch.load('test_cases/Ground_Truth/ground_truth_index_[786].pt')
    # ground_clas_output, ground_coord_output = rpn_net.create_ground_truth(truth['bboxes'], truth['index'], truth['grid_size'], truth['anchors'], truth['image_size'])
    # print("Ground Truth Class Test 1 - Number Wrong", torch.sum((truth['ground_clas'] != ground_clas_output).int()))
    # pmask = truth['ground_clas'] == 1
    # print("Ground Truth Coord Test 1 - Number Wrong", torch.all(torch.isclose(pmask * truth['ground_coord'], pmask * ground_coord_output.double(), .0001)))
    # rpn_net = RPNHead()
    # truth = torch.load('test_cases/Ground_Truth/ground_truth_index_[951].pt')
    # ground_clas_output, ground_coord_output = rpn_net.create_ground_truth(truth['bboxes'], truth['index'], truth['grid_size'], truth['anchors'], truth['image_size'])
    # print("Ground Truth Class Test 2 - Number Wrong", torch.sum((truth['ground_clas'] != ground_clas_output).int()))
    # pmask = truth['ground_clas'] == 1
    # print("Ground Truth Coord Test 2 - Number Wrong", torch.all(torch.isclose(pmask * truth['ground_coord'], pmask * ground_coord_output.double(), .0001)))
    # rpn_net = RPNHead()
    # truth = torch.load('test_cases/Ground_Truth/ground_truth_index_[1075].pt')
    # ground_clas_output, ground_coord_output = rpn_net.create_ground_truth(truth['bboxes'], truth['index'], truth['grid_size'], truth['anchors'], truth['image_size'])
    # print("Ground Truth Class Test 3 - Number Wrong", torch.sum((truth['ground_clas'] != ground_clas_output).int()))
    # pmask = truth['ground_clas'] == 1
    # print("Ground Truth Coord Test 3 - Number Wrong", torch.all(torch.isclose(pmask * truth['ground_coord'], pmask * ground_coord_output.double(), .0001)))
    # truth = torch.load('test_cases/Ground_Truth/ground_truth_index_[1358].pt')
    # ground_clas_output, ground_coord_output = rpn_net.create_ground_truth(truth['bboxes'], truth['index'], truth['grid_size'], truth['anchors'], truth['image_size'])
    # print("Ground Truth Class Test 4 - Number Wrong", torch.sum((truth['ground_clas'] != ground_clas_output).int()))
    # pmask = truth['ground_clas'] == 1
    # print("Ground Truth Coord Test 4 - Number Wrong", torch.all(torch.isclose(pmask * truth['ground_coord'], pmask * ground_coord_output.double(), .0001)))
    # truth = torch.load('test_cases/Ground_Truth/ground_truth_index_[1691].pt')
    # ground_clas_output, ground_coord_output = rpn_net.create_ground_truth(truth['bboxes'], truth['index'], truth['grid_size'], truth['anchors'], truth['image_size'])
    # print("Ground Truth Class Test 5 - Number Wrong", torch.sum((truth['ground_clas'] != ground_clas_output).int()))
    # pmask = truth['ground_clas'] == 1
    # print("Ground Truth Coord Test 5 - Number Wrong", torch.all(torch.isclose(pmask * truth['ground_coord'], pmask * ground_coord_output.double(), .0001)))

    # ###
    # # Loss Tests
    # ###
    # rpn_net = RPNHead()
    # truth = torch.load('test_cases/Loss/loss_test_0.pt')
    # loss_c_output = rpn_net.loss_class(truth['p_out'], truth['n_out'])
    # loss_r_output = rpn_net.loss_reg(truth['pos_target_coord'], truth['pos_out_r'])    
    # print("Loss Class Test 1 - ", torch.all(torch.isclose(truth['loss_c'], loss_c_output.float(), .1)))
    # print("Loss Regression Test 1 - ", torch.all(torch.isclose(truth['loss_r'], loss_r_output.float(), .0001)))
    # truth = torch.load('test_cases/Loss/loss_test_1.pt')
    # loss_c_output = rpn_net.loss_class(truth['p_out'], truth['n_out'])
    # loss_r_output = rpn_net.loss_reg(truth['pos_target_coord'], truth['pos_out_r'])    
    # print("Loss Class Test 2 - ", torch.all(torch.isclose(truth['loss_c'], loss_c_output.float(), .1)))
    # print("Loss Regression Test 2 - ", torch.all(torch.isclose(truth['loss_r'], loss_r_output.float(), .0001)))
    # truth = torch.load('test_cases/Loss/loss_test_2.pt')
    # loss_c_output = rpn_net.loss_class(truth['p_out'], truth['n_out'])
    # loss_r_output = rpn_net.loss_reg(truth['pos_target_coord'], truth['pos_out_r'])    
    # print("Loss Class Test 3 - ", torch.all(torch.isclose(truth['loss_c'], loss_c_output.float(), .1)))
    # print("Loss Regression Test 3 - ", torch.all(torch.isclose(truth['loss_r'], loss_r_output.float(), .0001)))
    # truth = torch.load('test_cases/Loss/loss_test_3.pt')
    # loss_c_output = rpn_net.loss_class(truth['p_out'], truth['n_out'])
    # loss_r_output = rpn_net.loss_reg(truth['pos_target_coord'], truth['pos_out_r'])    
    # print("Loss Class Test 4 - ", torch.all(torch.isclose(truth['loss_c'], loss_c_output.float(), .1)))
    # print("Loss Regression Test 4 - ", torch.all(torch.isclose(truth['loss_r'], loss_r_output.float(), .0001)))
    # truth = torch.load('test_cases/Loss/loss_test_4.pt')
    # loss_c_output = rpn_net.loss_class(truth['p_out'], truth['n_out'])
    # loss_r_output = rpn_net.loss_reg(truth['pos_target_coord'], truth['pos_out_r'])    
    # print("Loss Class Test 5 - ", torch.all(torch.isclose(truth['loss_c'], loss_c_output.float(), .1)))
    # print("Loss Regression Test 5 - ", torch.all(torch.isclose(truth['loss_r'], loss_r_output.float(), .0001)))
    # truth = torch.load('test_cases/Loss/loss_test_5.pt')
    # loss_c_output = rpn_net.loss_class(truth['p_out'], truth['n_out'])
    # loss_r_output = rpn_net.loss_reg(truth['pos_target_coord'], truth['pos_out_r'])    
    # print("Loss Class Test 6 - ", torch.all(torch.isclose(truth['loss_c'], loss_c_output.float(), .1)))
    # print("Loss Regression Test 6 - ", torch.all(torch.isclose(truth['loss_r'], loss_r_output.float(), .0001)))
    # truth = torch.load('test_cases/Loss/loss_test_6.pt')
    # loss_c_output = rpn_net.loss_class(truth['p_out'], truth['n_out'])
    # loss_r_output = rpn_net.loss_reg(truth['pos_target_coord'], truth['pos_out_r'])    
    # print("Loss Class Test 7 - ", torch.all(torch.isclose(truth['loss_c'], loss_c_output.float(), .1)))
    # print("Loss Regression Test 7 - ", torch.all(torch.isclose(truth['loss_r'], loss_r_output.float(), .0001)))
    # truth = torch.load('test_cases/Loss/loss_test_7.pt')
    # loss_c_output = rpn_net.loss_class(truth['p_out'], truth['n_out'])
    # loss_r_output = rpn_net.loss_reg(truth['pos_target_coord'], truth['pos_out_r'])    
    # print("Loss Class Test 8 - ", torch.all(torch.isclose(truth['loss_c'], loss_c_output.float(), .1)))
    # print("Loss Regression Test 8 - ", torch.all(torch.isclose(truth['loss_r'], loss_r_output.float(), .0001)))
    # truth = torch.load('test_cases/Loss/loss_test_8.pt')
    # loss_c_output = rpn_net.loss_class(truth['p_out'], truth['n_out'])
    # loss_r_output = rpn_net.loss_reg(truth['pos_target_coord'], truth['pos_out_r'])    
    # print("Loss Class Test 9 - ", torch.all(torch.isclose(truth['loss_c'], loss_c_output.float(), .1)))
    # print("Loss Regression Test 9 - ", torch.all(torch.isclose(truth['loss_r'], loss_r_output.float(), .0001)))
    # truth = torch.load('test_cases/Loss/loss_test_9.pt')
    # loss_c_output = rpn_net.loss_class(truth['p_out'], truth['n_out'])
    # loss_r_output = rpn_net.loss_reg(truth['pos_target_coord'], truth['pos_out_r'])    
    # print("Loss Class Test 10 - ", torch.all(torch.isclose(truth['loss_c'], loss_c_output.float(), .1)))
    # print("Loss Regression Test 10 - ", torch.all(torch.isclose(truth['loss_r'], loss_r_output.float(), .0001)))


    ###
    # Ground Truth Tests
    ###
    # box_head = BoxHead()
    # truth = torch.load('test_cases_b/GroundTruth/ground_truth_test0.pt')
    # labels, regressor_target = box_head.create_ground_truth(truth['proposals'], truth['gt_labels'], truth['bbox'])
    # print("Ground Truth Class Test 1 - Number Wrong", torch.sum((truth['labels'] != labels).int()))
    # pmask = truth['labels'] != 0
    # print("Ground Truth Coord Test 1 - Number Wrong", torch.all(torch.isclose(pmask * truth['regressor_target'], pmask * regressor_target.float(), .0001)))
    # box_head = BoxHead()
    # truth = torch.load('test_cases_b/GroundTruth/ground_truth_test1.pt')
    # labels, regressor_target = box_head.create_ground_truth(truth['proposals'], truth['gt_labels'], truth['bbox'])
    # print("Ground Truth Class Test 2 - Number Wrong", torch.sum((truth['labels'] != labels).int()))
    # pmask = truth['labels'] != 0
    # print("Ground Truth Coord Test 2 - Number Wrong", torch.all(torch.isclose(pmask * truth['regressor_target'], pmask * regressor_target.float(), .0001)))
    # box_head = BoxHead()
    # truth = torch.load('test_cases_b/GroundTruth/ground_truth_test2.pt')
    # labels, regressor_target = box_head.create_ground_truth(truth['proposals'], truth['gt_labels'], truth['bbox'])
    # print("Ground Truth Class Test 3 - Number Wrong", torch.sum((truth['labels'] != labels).int()))
    # pmask = truth['labels'] != 0
    # print("Ground Truth Coord Test 3 - Number Wrong", torch.all(torch.isclose(pmask * truth['regressor_target'], pmask * regressor_target.float(), .0001)))
    box_head = BoxHead()
    truth = torch.load('test_cases_b/GroundTruth/ground_truth_test3.pt')
    labels, regressor_target = box_head.create_ground_truth(truth['proposals'], truth['gt_labels'], truth['bbox'])
    print("Ground Truth Class Test 4 - Number Wrong", torch.sum((truth['labels'] != labels).int()))
    pmask = truth['labels'] != 0
    print("Ground Truth Coord Test 4 - Number Wrong", torch.all(torch.isclose(pmask * truth['regressor_target'], pmask * regressor_target.float(), .0001)))
    print(truth['labels'], labels)
    # box_head = BoxHead()
    # truth = torch.load('test_cases_b/GroundTruth/ground_truth_test4.pt')
    # labels, regressor_target = box_head.create_ground_truth(truth['proposals'], truth['gt_labels'], truth['bbox'])
    # print("Ground Truth Class Test 5 - Number Wrong", torch.sum((truth['labels'] != labels).int()))
    # pmask = truth['labels'] != 0
    # print("Ground Truth Coord Test 5 - Number Wrong", torch.all(torch.isclose(pmask * truth['regressor_target'], pmask * regressor_target.float(), .0001)))
    # box_head = BoxHead()
    # truth = torch.load('test_cases_b/GroundTruth/ground_truth_test5.pt')
    # labels, regressor_target = box_head.create_ground_truth(truth['proposals'], truth['gt_labels'], truth['bbox'])
    # print("Ground Truth Class Test 6 - Number Wrong", torch.sum((truth['labels'] != labels).int()))
    # pmask = truth['labels'] != 0
    # print("Ground Truth Coord Test 6 - Number Wrong", torch.all(torch.isclose(pmask * truth['regressor_target'], pmask * regressor_target.float(), .0001)))
    # box_head = BoxHead()
    # truth = torch.load('test_cases_b/GroundTruth/ground_truth_test6.pt')
    # labels, regressor_target = box_head.create_ground_truth(truth['proposals'], truth['gt_labels'], truth['bbox'])
    # print("Ground Truth Class Test 7 - Number Wrong", torch.sum((truth['labels'] != labels).int()))
    # pmask = truth['labels'] != 0
    # print("Ground Truth Coord Test 7 - Number Wrong", torch.all(torch.isclose(pmask * truth['regressor_target'], pmask * regressor_target.float(), .0001)))