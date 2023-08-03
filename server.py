from typing import Dict, Optional, Tuple
from collections import OrderedDict
import argparse
from torch.utils.data import DataLoader
import utils
import numpy

import flwr as fl
import torch

import utils

import warnings

warnings.filterwarnings("ignore")

def fit_config(server_round: int):
	"""Return training configuration dict for each round."""
	config = {
	"batch_size": 12,
	"local_epochs": 5,
	"learning_rate": 0.002,
	"num_workers": 1
	}
	return config

def evaluate_config(server_round: int):
	"""Return evaluation configuration dict for each round.
	Perform how many batches will be utilized on local evaluation steps on each client (i.e., use five
	batches)."""
	val_steps = 313 # For main model len(validset)/batch_size
	return {"val_steps": val_steps}

def get_evaluate_fn(model: torch.nn.Module):
	"""Return an evaluation function for server-side evaluation."""

	# Load data and model here to avoid the overhead of doing it in `evaluate` itself
	_, valset, _ = utils.load_data()
	valLoader = DataLoader(valset, batch_size=16, collate_fn=utils.collate_fn, num_workers=1)

	# The `evaluate` function will be called after every round
	def evaluate(
		server_round: int,
		parameters: fl.common.NDArrays,
		config: Dict[str, fl.common.Scalar],
	    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
		# Update model with the latest parameters
		params_dict = zip(model.state_dict().keys(), parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		model.load_state_dict(state_dict, strict=True)
		
		device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

		__, metrics = utils.test(model, valLoader, _, device)

		return __, {"metrics" : metrics}

	return evaluate

def main():
	"""Load model for
	1. server-side parameter initialization
	2. server-side parameter evaluation
	"""
	parser = argparse.ArgumentParser(description="Flower")
	parser.add_argument(
	"--clientnumber",
	type=int,
	default=2,
	choices=range(1, 7),
	required=False	,
	help="Specifies the client number to be used. \
	Picks 2 client by default",
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

	model = utils.load_net(args.model)

	model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

	# Create strategy
	strategy = fl.server.strategy.FedAvg(
		fraction_fit=1.0, #sample all of available clients for training
		fraction_evaluate=1.0, #sample all of available clients for evaluation
		min_fit_clients=args.clientnumber, #Never sample less than number of clients for training
		min_evaluate_clients=args.clientnumber, #Never sample less than number of clients for evaluation
		min_available_clients=args.clientnumber, #Wait until all number of clients are available
		evaluate_fn=get_evaluate_fn(model),
		on_fit_config_fn=fit_config,
		on_evaluate_config_fn=evaluate_config,
		initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
		)

	# Start Flower server for ### rounds of federated learning
	fl.server.start_server(
	server_address="localhost:8080",
	config=fl.server.ServerConfig(num_rounds=3),
	strategy=strategy
	)

if __name__ == "__main__":
	main()
