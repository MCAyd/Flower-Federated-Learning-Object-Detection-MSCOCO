import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2 as fasterrcnn, FasterRCNN_ResNet50_FPN_V2_Weights as fasterrcnn_weights
from torchvision.models.detection import fcos_resnet50_fpn as fcos, FCOS_ResNet50_FPN_Weights as fcos_weights
from torchvision.models.detection import retinanet_resnet50_fpn_v2 as retinanet, RetinaNet_ResNet50_FPN_V2_Weights as retinanet_weights
from torchvision.models.detection import ssd300_vgg16 as ssd, SSD300_VGG16_Weights as ssd_weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large as ssdlite, SSDLite320_MobileNet_V3_Large_Weights as ssdlite_weights
from torchvision.models import ResNet50_Weights as resnet50_weights
from coco_transfer import CocoDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy
import random

import os
import os.path
import json
import uuid

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(4)
random.seed(4)
numpy.random.seed(4)

train_data_dir = '/scratch/dataset/coco/images/train2017'
train_ann_file = '/scratch/dataset/coco/annotations/instances_train2017.json'
val_data_dir = '/scratch/dataset/coco/images/val2017'
val_ann_file = '/scratch/dataset/coco/annotations/instances_val2017.json'
result_path = '/scratch/dataset/coco/annotations' 
eval_thrs = 0.25

def load_data():
	"""Load COCO set."""
	transform = transforms.Compose(
			[
				transforms.ToTensor(),
			]
		)

	trainset = CocoDetection(root=train_data_dir, annotation=train_ann_file, transforms=transform)
	validset = CocoDetection(root=val_data_dir, annotation=val_ann_file, transforms=transform)
	
	#trainset = torch.utils.data.Subset(trainset, range(1000)) # TOY_MODEL
	#validset = torch.utils.data.Subset(validset, range(100))

	num_examples = {"trainset": len(trainset), "validset": len(validset)}

	return trainset, validset, num_examples

def load_partition(idx: int, cnumber: int, iid: bool = True):
	"""Load 1/client_number th of the training and test data to simulate a partition."""
	assert idx in range(3)
	trainset, validset, num_examples = load_data()
	
	if iid == False:
		trainset = sorted(trainset, key=lambda x: len(x[2]["labels"]), reverse=True)
		
	n_train = int(num_examples["trainset"] / cnumber)
	n_valid = int(num_examples["validset"] / cnumber)

	train_partition = torch.utils.data.Subset(
	trainset, range(idx * n_train, (idx + 1) * n_train)
	)
	valid_partition = torch.utils.data.Subset(
	validset, range(idx * n_valid, (idx + 1) * n_valid)
	)

	return (train_partition,valid_partition)

def load_net(entrypoint: str = 'none', pretrained: bool = False):
	"""Load net."""
	print("Object detection model is loaded: " + entrypoint + ". Pretrained model: " + str(pretrained))
	if entrypoint == "fasterrcnn":
		if pretrained:
			model = fasterrcnn(weights=fasterrcnn_weights)
		else:
			model = fasterrcnn(weights_backbone=resnet50_weights) #weights_backbone set None as default, manually imported
	elif entrypoint == "fcos":
		if pretrained:
			model = fcos(weights=fcos_weights)
		else:
			model = fcos()
	elif entrypoint == "retinanet":
		if pretrained:
			model = retinanet(weights=retinanet_weights)
		else:
			model = retinanet(weights_backbone=resnet50_weights) #weights_backbone set None as default, manually imported
	elif entrypoint == "ssd":
		if pretrained:
			model = ssd(weights=ssd_weights)
		else:
			model = ssd()
	else:
		if pretrained:
			model = ssdlite(weights=ssdlite_weights)
		else:
			model = ssdlite()
			
	return model

def get_model_params(model):
	"""Returns a model's parameters."""
	return [val.cpu().numpy() for _, val in model.state_dict().items()]
	
# collate_fn needs for batch
def collate_fn(batch):
	return tuple(zip(*batch))

def train(net, trainloader, valloader, epochs, lrate, momentum, weight_decay, server_round, client_no, device: str = "cpu"):
	"""Train the network on the training set."""
	print("Starting training in device, " + str(device) + '...')
	
	net.to(device)  # move model to GPU if available
	optimizer = torch.optim.SGD(
	net.parameters(), lr=lrate, momentum=momentum, weight_decay=weight_decay)
	scaler = torch.cuda.amp.GradScaler()
	lr_scheduler = None
	if server_round == 5:
		lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4], gamma=0.1)

	train_results = []  # initialize a list to store training results for each epoch
	for epoch in range(epochs):
	
		ovr_loss, batch_ = train_one_epoch(net, optimizer, trainloader, device, epoch, server_round, scaler)
		if lr_scheduler is not None:
			lr_scheduler.step()
			
		# calculate average loss and perplexity for this epoch
		avg_loss = ovr_loss / batch_
		perplexity = torch.exp(avg_loss)

		# store results for this epoch in the dictionary
		train_results.append(str
			('client: ' + client_no +
			'epoch: ' + str(epoch) +
			' avg_loss: ' + str(avg_loss.item()) +  # ensure the value is a standard python number, not a tensor
			' perplexity:' + str(perplexity.item())))

		print('Client {}, Epoch {}, Loss: {:.4f}, Perplexity: {:5.4f}'
		      .format(client_no, epoch, avg_loss.item(), perplexity.item())) 

	net.to("cpu")  # move model back to CPU
	
	train_results = ' '.join(result for result in train_results)

	return {"round results": train_results }

def test(net, testloader, steps: int = None, device: str = 'cpu'):
	"""Validate the network on the entire test set."""
	print("Starting evaluation in device, " + str(device) + '...')
	net.to(device)  # move model to GPU if available

	predictions = []
		
	net.eval()
	with torch.no_grad():
		for batch_idx, (image_ids, images, targets) in enumerate(testloader):
			image_ids = list(image_id for image_id in image_ids)
			images = list(image.to(device) for image in images)
			model_predictions = net(images) # gives boxes, classes, and score predictions while net in eval.
			model_predictions = [{k: v.cpu().tolist() for k, v in prediction_dict.items()} for prediction_dict in model_predictions]
			converted_results = coco_convert(image_ids, model_predictions)
			predictions.extend(converted_results)
			if steps is not None and batch_idx == steps:
				break	
	net.to("cpu")  # move model back to CPU
	loss = None
	
	if predictions == []:
		eval_error = 'The model cannot return prediction yet, requires more training'
		return loss, eval_error
		
	random_name = str(uuid.uuid4())
	# Define the path
	json_file_path = os.path.join(result_path, f"{random_name}.json")
	
	#print(predictions) # check format
	# Write predictions to the json file
	with open(json_file_path, 'w') as f:
		json.dump(predictions, f)
		
	metrics = coco_evaluation(val_ann_file, json_file_path)
	
	metrics_string = numpy.array2string(metrics, separator=',', formatter={'float_kind':lambda x: "%.4f" % x})
	
	print("Evaluation metrics, " + metrics_string)
	
	# Delete the json file, calculations are already made
	os.remove(json_file_path)
	
	return loss, metrics_string
	
def coco_evaluation(gts, preds, iou_type="bbox"):
	"""
	gts: Ground truth annotations
	preds: Prediction annotations
	iou_type: Type of IoU, e.g., 'bbox', 'segm', 'keypoints'
	"""
	cocoGT = COCO(gts)  # Load the ground truth annotations
	cocoDT = cocoGT.loadRes(preds)  # Load the prediction annotations

	coco_eval = COCOeval(cocoGT, cocoDT, iou_type)
	#coco_eval.params.recThrs = eval_thrs
	
	coco_eval.evaluate()
	coco_eval.accumulate()
	coco_eval.summarize()

	return coco_eval.stats
	
def coco_convert(img_ids,preds):
	batch_results = []
	for i, img_id in enumerate(img_ids):
		size = len(preds[i]['boxes'])
		for item in range(0,size):
			image_id = img_id.item()
			category_id = preds[i]['labels'][item]
			#[x1, y1, x2, y2]
			box = bbox = preds[i]['boxes'][item]
			bbox[2] = abs(box[2]-box[0])
			bbox[3]= abs(box[3]-box[1])
			bbox_coords = [round(coord, 1) for coord in bbox]
			score = preds[i]['scores'][item]
			result = {"image_id":image_id,
				"category_id":category_id,
				"bbox":bbox_coords, #[x,y,width,height]
				"score":score}
				
			#if result["score"] >= eval_thrs:
				#print(result) # check format
			batch_results.append(result)
			
	return batch_results
	
def train_one_epoch(model, optimizer, trainloader, device, epoch, server_round, scaler=None):
    model.train()
    ovr_loss = 0
    batch_ = 0
    lr_scheduler = None
    
    if epoch == 0 and server_round == 1: # only in first round, epoch 0
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(trainloader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for _, images, targets in trainloader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            ovr_loss += losses
            batch_+=1

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

    return ovr_loss, batch_
			
