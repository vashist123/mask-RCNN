import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from utils_new import *
import matplotlib.pyplot as plt
from rpn_new import *
import matplotlib.patches as patches


#################
# Detect CoLab
#################
try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
      images_h5 = h5py.File(paths[0], mode = 'r')
      masks_h5 = h5py.File(paths[1], mode = 'r')
      self.images = images_h5.get(f'{list(images_h5.keys())[0]}')[()]
      self.masks = masks_h5.get(f'{list(masks_h5.keys())[0]}')[()]
      self.labels = np.load(paths[2],allow_pickle=True)
      self.bboxes = np.load(paths[3],allow_pickle=True)
      self.new_masks = []
      j=0
      for i in range(len(self.labels)):
        self.new_masks.append(self.masks[j:j+len(self.labels[i])])
        j = j + len(self.labels[i])
        
        #############################################
        # TODO Initialize  Dataset
        #############################################


    # In this function for given index we rescale the image and the corresponding  masks, boxes
    # and we return them as output
    # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox
        # index
    def __getitem__(self, index):
        ################################
        # TODO return transformed images,labels,masks,boxes,index
        ################################


        img = self.images[index]
        mask = self.new_masks[index]
        label = self.labels[index]
        bbox = self.bboxes[index]

        transed_img,transed_mask,transed_bbox = self.pre_process_batch(img,mask,bbox)

        assert transed_img.shape == (3,800,1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]

        
        return transed_img, label, transed_mask, transed_bbox, index



    # This function preprocess the given image, mask, box by rescaling them appropriately
    # output:
    #        img: (3,800,1088)
    #        mask: (n_box,800,1088)
    #        box: (n_box,4)
    def pre_process_batch(self, img, mask, bbox):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.totensor = transforms.ToTensor()
        img = img.astype('float32')
        img = self.totensor(img)
        img = img / 255.0
        img = img
        img = img.permute(1,2,0)
        img = img.unsqueeze(dim=0)
        img = F.interpolate(img,(800,1066),mode = 'bilinear',align_corners = False)
        img = img.squeeze(dim=0)
        img = transforms.functional.normalize(img,(0.485,0.456,0.406),(0.229,0.224,0.225))
        img = F.pad(img,(11,11), 'constant', 0)

        mask = mask.astype('float32')
        mask = self.totensor(mask)
        mask = mask
        mask = mask.permute(1,2,0)
        mask = mask.unsqueeze(dim=0)
        mask = F.interpolate(mask,(800,1066), mode = 'bilinear', align_corners=False)
        mask = mask.squeeze(dim=0)
        mask = F.pad(mask,(11,11))
        bbox1 = torch.zeros(bbox.shape)
        bbox = torch.from_numpy(bbox)
        bbox1[:,0] = bbox[:,0] *800/300
        bbox1[:,1] = bbox[:,1] *1066/400 + 11
        bbox1[:,2] = bbox[:,2] *800/300
        bbox1[:,3] = bbox[:,3] *1066/400 + 11

        assert img.shape == (3, 800, 1088)
        assert bbox1.shape[0] == mask.shape[0]

        return img, mask, bbox1    

    
    def __len__(self):
        return len(self.images)




class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers


    # output:
    #  dict{images: (bz, 3, 800, 1088)
    #       labels: list:len(bz)
    #       masks: list:len(bz){(n_obj, 800,1088)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #       index: list:len(bz)
    def collect_fn(self, batch):
        transed_img_list = []
        label_list = []
        transed_mask_list = []
        transed_bbox_list = []
        index_list = []
        for i,(transed_img,label,transed_mask,transed_bbox,index) in enumerate(batch,0):
            transed_img_list.append(transed_img)
            label_list.append(label)
            transed_mask_list.append(transed_mask)
            transed_bbox_list.append(transed_bbox)
            index_list.append(index)
        out_batch = {"images": torch.stack(transed_img_list,dim =0 ), "labels": label_list , "masks":transed_mask_list,"bbox":transed_bbox_list,"index":index_list}
        return out_batch


    def loader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          collate_fn=self.collect_fn)


if __name__ == '__main__':
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
    batch_size = 1
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    mask_color_list = ["jet", "ocean", "Spectral"]
    class_list = ["vehicle","human","animal"]
    
    for iter,batch in enumerate(train_loader,0):
        images=batch['images'][0,:,:,:]
        indexes=batch['index']
        boxes=batch['bbox']
        mask = batch['masks']
        label = batch['labels']

        # print(images.shape)
        # print(len(indexes),indexes)
        # print(len(boxes),boxes[0].shape)
        # print(len(mask),mask[0].shape)
        # print(len(label),label[0].shape)

        for i in range(batch_size):
            labels_for_img = label[i]
            plot_img = images
            plot_img = transforms.functional.normalize(plot_img,(-0.485/0.229,-0.456/0.224,-0.406/0.225),(1/0.229,1/0.224,1/0.225))
            plot_img = plot_img.permute(1,2,0)
            plt.imshow(plot_img)

            masks = mask[i]
            mask_num = mask[i].shape[0]

            if mask_num == 1:
                print(labels_for_img)
                print(mask[i].shape)
                mask_init = torch.squeeze(mask[i])
                print(mask_init.shape)
                mask_new = np.ma.masked_where(mask_init==0,mask_init)
                plt.imshow(mask_new,cmap=mask_color_list[labels_for_img[0]-1], alpha=0.5)

            else:
                for j in range(mask_num):
                    label_for_box = labels_for_img[j]
                    mask_init = torch.squeeze(mask[i][j])
                    mask_new = np.ma.masked_where(mask_init==0,mask_init)
                    plt.imshow(mask_new,cmap=mask_color_list[label_for_box-1], alpha=0.5)

            ax = plt.gca()

            for j in range(len(label[i])):
                rect = patches.Rectangle((boxes[i][j][0],boxes[i][j][1]),boxes[i][j][2] - boxes[i][j][0],boxes[i][j][3] - boxes[i][j][1],linewidth=2,edgecolor='r',facecolor='none',label = "test")
                ax.add_patch(rect)
                plt.text(boxes[i][j][0],boxes[i][j][1],class_list[labels_for_img[j]-1],color = 'r')

            plt.show()







        # gt,ground_coord=rpn_net.create_batch_truth(boxes,indexes,images.shape[-2:])


        # # Flatten the ground truth and the anchors
        # flatten_coord,flatten_gt,flatten_anchors=output_flattening(ground_coord,gt,rpn_net.get_anchors())
        
        # # Decode the ground truth box to get the upper left and lower right corners of the ground truth boxes
        # decoded_coord=output_decoding(flatten_coord,flatten_anchors)
        
        # # Plot the image and the anchor boxes with the positive labels and their corresponding ground truth box
        # images = transforms.functional.normalize(images,
        #                                               [-0.485/0.229, -0.456/0.224, -0.406/0.225],
        #                                               [1/0.229, 1/0.224, 1/0.225], inplace=False)
        # fig,ax=plt.subplots(1,1)
        # ax.imshow(images.permute(1,2,0))
        
        # find_cor=(flatten_gt==1).nonzero()
        # find_neg=(flatten_gt==-1).nonzero()
             
        # for elem in find_cor:
        #     coord=decoded_coord[elem,:].view(-1)
        #     anchor=flatten_anchors[elem,:].view(-1)

        #     col='r'
        #     rect=patches.Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],fill=False,color=col)
        #     ax.add_patch(rect)
        #     rect=patches.Rectangle((anchor[0]-anchor[2]/2,anchor[1]-anchor[3]/2),anchor[2],anchor[3],fill=False,color='b')
        #     ax.add_patch(rect)

        # plt.show()
 
        # if(i>20):
        #     break
        

 