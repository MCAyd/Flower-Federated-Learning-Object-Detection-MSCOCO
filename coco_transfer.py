import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO

class CocoDetection(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        if num_objs == 0:
    	    boxes = torch.zeros((0, 4), dtype=torch.float32)
    	    areas = 0
    
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            if xmin != xmax and ymin!= ymax:
            	boxes.append([xmin, ymin, xmax, ymax])
            	area = coco_annotation[i]['bbox'][2] * coco_annotation[i]['bbox'][3]
            	areas.append(area)
            	#areas.append(coco_annotation[i]['area'])
            	labels.append(coco_annotation[i]['category_id'])
            	iscrowd.append(coco_annotation[i]['iscrowd'])

        # Tensorise img_id
        img_id = torch.tensor([img_id], dtype=torch.int64)
        # Size of bbox (Rectangular)
        boxes = torch.as_tensor(boxes, dtype=torch.float)
	# tensorise areas
        areas = torch.as_tensor(areas)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # iscrowd
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.uint8)
        
        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)
            
        #if my_annotation["image_id"] == 29187: # See if the format is correct
        	#print(my_annotation)

        return img_id, img, my_annotation

    def __len__(self):
        return len(self.ids)
