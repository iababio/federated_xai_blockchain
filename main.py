"""
Federated learning simulation with CIFAR-10 dataset using XAI and Flower framework.

This script sets up a federated learning environment with multiple clients and
uses a custom federated strategy for model training and evaluation. Visualization
functions analyze and display training metrics.
"""

from contextvars import Context
import flwr as fl
from src.client import XAIFederatedClient
from src.model import Net
from src.strategy import XAIFederatedStrategy
from src.utils import DEVICE, get_parameters, load_data
from src.visualization import analyze_results, plot_federated_learning_metrics

# Load and preprocess the CIFAR-10 dataset
trainloaders, valloaders, testloader = load_data()
NUM_CLIENTS = 10

client_ids = [str(i) for i in range(NUM_CLIENTS)]


def client_fn(context: Context) -> XAIFederatedClient:
    """
    Initializes a federated client with specific data loaders based on the client ID.

    Args:
        context (Context): Flower context object providing client node ID information.

    Returns:
        XAIFederatedClient: A federated client instance with assigned data loaders and model.
    """
    cid = int(context.node_id) % len(trainloaders)
    net = Net().to(DEVICE)

    if cid < len(trainloaders):
        trainloader = trainloaders[cid]
        valloader = valloaders[cid]
    else:
        raise IndexError(
            f"Client ID {cid} exceeds the number of available data loaders."
        )

    # Ensure the model parameters are correctly initialized
    initial_parameters = get_parameters(net)

    client = XAIFederatedClient(cid, net, trainloader, valloader)

    # Set the initial parameters for the client
    client.set_parameters(initial_parameters)

    return client


# Modify the FedSparse strategy to include the evaluation function
strategy = XAIFederatedStrategy()

# Start the federated learning simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
    client_resources={"num_cpus": 1, "num_gpus": 0},
)

# Analyze and visualize results
analyze_results(strategy)
plot_federated_learning_metrics(strategy)
