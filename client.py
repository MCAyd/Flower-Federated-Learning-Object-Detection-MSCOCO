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
		device: str,
		client_no: str):
		
		self.model = model
		self.device = device
		self.client_no = client_no
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
		self.momentum: float = config['momentum']
		self.weight_decay: float = config['weight_decay']
		self.server_round: int = config['server_round']

		# Update local model parameters
		self.set_parameters(parameters)

		trainLoader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, collate_fn=utils.collate_fn, num_workers=self.num_workers)

		results = utils.train(self.model, trainLoader, None, self.epochs, self.lrate, self.momentum, self.weight_decay, self.server_round, self.client_no, self.device)

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
	
def client_dry_run(model, pretrained, noniid, device: str = "cpu"):
	"""Tests to check whether all client methods are working as
	expected AND evaluate pretrained model as for one client."""
	trainset, validset = utils.load_partition(0, 1, noniid)
	#trainset = torch.utils.data.Subset(trainset, range(1000)) # dry_toy_run
	#validset = torch.utils.data.Subset(validset, range(100))
	client = CocoClient(model, trainset, validset, device, str(0))
	
	config = {"batch_size": 16,
		"local_epochs": 6,
		"learning_rate": 0.005,
		"num_workers": 0,
		"momentum": 0.9,
		"weight_decay": 1e-4,
		"server_round": 0}
		
	if pretrained != True:
		parameters_trained, _, _=client.fit(
		utils.get_model_params(model), config)

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
	default=1,
	choices=range(1, 5),
	required=False	,
	help="Specifies the client number to be used. \
	Picks 1 client by default",
	)
	parser.add_argument(
	"--partition",
	type=int,
	default=0,
	choices=range(0, 4),
	required=False,
	help="Specifies the artificial data partition of MSCOCO to be used. \
	Picks partition 0 by default",
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
	"--noniid",
	type=bool,
	default=False,
	required=False,
	help="Set to true for noniid partition. Default: False",
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
        	client_dry_run(model, args.pretrained, args.noniid, device)
        
	else:
		trainset,validset = utils.load_partition(args.partition, args.clientnumber, args.noniid)

		client = CocoClient(model, trainset, validset, device, str(args.partition))

		fl.client.start_numpy_client(server_address="localhost:8080", client=client)


if __name__ == "__main__":
	main()
