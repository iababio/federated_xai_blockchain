from contextvars import Context
import flwr as fl
from client import XAIFederatedClient
from model import Net
from strategy import XAIFederatedStrategy
from utils import DEVICE, get_parameters, load_data
from visualization import analyze_results

# Load and preprocess the CIFAR-10 dataset
trainloaders, valloaders, testloader = load_data()
num_clients = 10


client_ids = [str(i) for i in range(num_clients)]

def client_fn(context: Context) -> XAIFederatedClient:
    cid = int(context.node_id) % len(trainloaders)
    net = Net().to(DEVICE)

    if cid < len(trainloaders):
        trainloader = trainloaders[cid]
        valloader = valloaders[cid]
    else:
        raise IndexError(f"Client ID {cid} exceeds the number of available data loaders.")

    # Ensure the model parameters are correctly initialized
    initial_parameters = get_parameters(net)

    client = XAIFederatedClient(cid, net, trainloader, valloader)

    # Set the initial parameters for the client
    client.set_parameters(initial_parameters)

    return client



# Modify the FedSparse strategy to include the evaluation function
strategy = XAIFederatedStrategy()

# Continue with the simulation setup


fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=num_clients,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
    client_resources={"num_cpus": 1, "num_gpus": 0}
)

# Analyze and visualize results
analyze_results(strategy)