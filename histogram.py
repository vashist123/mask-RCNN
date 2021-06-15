from dataset import *
import math

#################
# Detect CoLab
#################
try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

def get_box_prop(bbox):

	ar = []
	sc = []

	for box in bbox:

		x1 = box[0]
		y1 = box[1]
		x2 = box[2]
		y2 = box[3]

		h = y2 - y1
		w = x2 - x1
		sc.append(math.sqrt(w*h))
		ar.append((w/(h+0.0000000000001)).item())

	return  sc,ar

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
    # rpn_net = RPNHead()
    # push the randomized training data into the dataloader

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_size = 1
    build_loader = BuildDataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    loader = build_loader.loader()

    scales = []
    aspect_ratios = []
    

    ## appending the scales and aspect ratios to a list 

    for i,batch in enumerate(loader,0):
        images=batch['images'][0,:,:,:]
        indexes=batch['index']
        boxes=batch['bbox']
        scale,aspect_ratio = get_box_prop(boxes[0])
        scales.extend(scale)
        aspect_ratios.extend(aspect_ratio)

    ##plotting the histograms in 50 bins

    plt.hist(scales,bins= 50)
    plt.xlabel('scale')
    plt.show()
    plt.hist(aspect_ratios,bins= 50)
    plt.xlabel('aspect ratios')
    plt.show()
