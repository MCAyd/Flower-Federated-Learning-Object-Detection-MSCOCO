# Flower-Federated-Learning-Object-Detection-MSCOCO
This project demonstrates an advanced federated learning setup using Flower with PyTorch for object detection model. The project has key aspects in the following ways:

- You can choose the client number and the object detection model you would like to train/run from run.sh doc. 
- Each client holds a partition of MS COCO dataset of 118k training examples and 5k validation examples (note that using the `run.sh` script will select 3 clients with 3 partition of the dataset and object detection model ssdlite).
- Server-side model evaluation after parameter aggregation
- Hyperparameter schedule using config functions
- Custom return values
- Server-side parameter initialization

### Installing Dependencies

Project dependencies (such as `torch` and `flwr`) are defined in `pyproject.toml` and `requirements.txt`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)) or [pip](https://pip.pypa.io/en/latest/development/), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

#### Poetry

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
poetry run python3 -c "import flwr"
```

If you don't see any errors you're good to go!

#### pip

Write the command below in your terminal to install the dependencies according to the configuration file requirements.txt.

```shell
pip install -r requirements.txt
```

## Run Federated Learning with PyTorch and Flower

The included `run.sh` will start the Flower server (using `server.py`),
sleep for 5 seconds to ensure that the server is up, and then start 2 Flower clients (using `client.py`) with 2 partitions of the data.

```shell
poetry run ./run.sh
```

The `run.sh` script starts processes in the background so that you don't have to open eleven terminal windows. If you experiment with the code example and something goes wrong, simply using `CTRL + C` on Linux (or `CMD + C` on macOS) wouldn't normally kill all these processes, which is why the script ends with `trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT` and `wait`. This simply allows you to stop the experiment using `CTRL + C` (or `CMD + C`). If you change the script and anything goes wrong you can still use `killall python` (or `killall python3`) to kill all background processes (or a more specific command if you have other Python processes running that you don't want to kill).

You can also manually run `poetry run python3 server.py` and `poetry run python3 client.py` for as many clients as you want but you have to make sure that each command is ran in a different terminal window (or a different computer on the network).

## Run Dry Run Setting with Pretrained Models.

This setting allows only getting evaluation performance of a pretrained model while client is in dry run setting. In this setting, no data partition is made and only one client is operated with all validation dataset where the dataset is utilized for evaluation of the pretrained model.

```shell
poetry run python3 client.py --pretrained True --dry True --model fasterrcnn
```

To check whether clients are working properly and debug the client class with a small subset of the data;

```shell
poetry run python3 client.py --dry True --model fasterrcnn
```
