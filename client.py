import utils
from torch.utils.data import DataLoader
import torchvision.datasets
import torch
import numpy
import flwr as fl
import argparse
from collections import OrderedDict
import random

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(4)
random.seed(4)
numpy.random.seed(4)

class CocoClient(fl.client.NumPyClient):
	def __init__(
		self,
		model,
		trainset: torchvision.datasets,
		validset: torchvision.datasets,
		device: str):
		self.model = model
		self.device = device
		self.trainset = trainset
		self.validset = validset
		
	def set_parameters(self, parameters):
		"""Loads a net model for the client."""
		#model = utils.load_net()
		params_dict = zip(self.model.state_dict().keys(), parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.model.load_state_dict(state_dict, strict=True)

		#return self.model

	def fit(self, parameters, config):
		"""Train parameters on the locally held training set."""
		# Get hyperparameters for this round
		self.batch_size: int = config["batch_size"]
		self.epochs: int = config["local_epochs"]
		self.lrate: float = config['learning_rate']
		self.num_workers: int = config['num_workers']

		# Update local model parameters
		self.set_parameters(parameters)

		trainLoader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, collate_fn=utils.collate_fn, num_workers=self.num_workers)

		results = utils.train(self.model, trainLoader, None, self.epochs, self.lrate, self.device)

		parameters_prime = utils.get_model_params(self.model)
		num_examples_train = len(self.trainset)

		return parameters_prime, num_examples_train, results
		
	def evaluate(self, parameters, config):
		"""Evaluate parameters on the locally held test set."""
		# Update local model parameters
		self.set_parameters(parameters)
		
		if len(config) > 1: # the case where dry run is operated.
			self.batch_size = config["batch_size"]
			self.num_workers = config["num_workers"]
		
		valLoader = DataLoader(self.validset, batch_size=self.batch_size, collate_fn=utils.collate_fn, num_workers=self.num_workers)

		# Get config values
		steps: int = config["val_steps"]

		_, metrics = utils.test(self.model, valLoader, steps, self.device)
		
		return 0.0, len(self.validset), {"metrics" : metrics}
	
def client_dry_run(model, pretrained, device: str = "cpu"):
	"""Tests to check whether all client methods are working as
	expected AND evaluate pretrained model as for one client."""
	trainset, validset = utils.load_partition(0, 1)
	#trainset = torch.utils.data.Subset(trainset, range(10)) # dry_toy_run
	#validset = torch.utils.data.Subset(validset, range(10))
	client = CocoClient(model, trainset, validset, device)
	
	config = {"batch_size": 16,
		"local_epochs": 5,
		"learning_rate": 0.001,
		"num_workers": 1}
		
	if pretrained != True:
		parameters_trained, _, _=client.fit(
		utils.get_model_params(model),
		{"batch_size":config["batch_size"], "local_epochs":config["local_epochs"], "learning_rate":config["learning_rate"], "num_workers":config["num_workers"]},
		)

		client.evaluate(parameters_trained, {"val_steps":numpy.ceil(len(validset)/config["batch_size"])})
	else:
		client.evaluate(utils.get_model_params(model), {"val_steps":numpy.ceil(len(validset)/config["batch_size"]), "batch_size":config["batch_size"], "num_workers":config["num_workers"]})

	print("Dry Run Successful")

def main() -> None:
	parser = argparse.ArgumentParser(description="Flower")
	parser.add_argument(
	"--dry",
	type=bool,
	default=False,
	required=False,
	help="Do a dry-run to check the client",
	)
	parser.add_argument(
	"--clientnumber",
	type=int,
	default=2,
	choices=range(1, 4),
	required=False	,
	help="Specifies the client number to be used. \
	Picks 2 client by default",
	)
	parser.add_argument(
	"--partition",
	type=int,
	default=1,
	choices=range(0, 3),
	required=False,
	help="Specifies the artificial data partition of MSCOCO to be used. \
	Picks partition 1 by default",
	)
	parser.add_argument(
	"--use_cuda",
	type=bool,
	default=True,
	required=False,
	help="Set to true to use GPU. Default: True",
	)
	parser.add_argument(
	"--pretrained",
	type=bool,
	default=False,
	required=False,
	help="Set to true to use pretrained model. Default: False, Only available in dry_run setting",
	)
	parser.add_argument(
	"--model",
	type=str,
	default='fasterrcnn',
	choices=('fasterrcnn','fcos','retinanet','ssd','ssdlite'),
	required=False,
	help="Choose the torch model to be used, fasterrcnn as default",
	)

	args = parser.parse_args()

	device = torch.device(
	"cuda:"+str(args.partition) if torch.cuda.is_available() and args.use_cuda else "cpu"
	)
	
	model = utils.load_net(args.model, args.pretrained)
	
	if args.dry:
        	client_dry_run(model, args.pretrained, device)
        
	else:
		trainset,validset = utils.load_partition(args.partition, args.clientnumber)

		client = CocoClient(model, trainset, validset, device)

		fl.client.start_numpy_client(server_address="localhost:8080", client=client)


if __name__ == "__main__":
	main()
