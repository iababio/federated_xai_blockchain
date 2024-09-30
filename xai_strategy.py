
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import shap
from typing import List, Tuple, Union, Optional, Dict
from flwr.common import Parameters, Scalar
class XAIFederatedStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Apply causal insights and counterfactual scenarios to optimize federation
            optimized_parameters = self.optimize_federation(aggregated_parameters)
            return optimized_parameters, metrics

        return aggregated_parameters, metrics

    def optimize_federation(self, parameters: Parameters) -> Parameters:
        # Example: Adjust parameters based on causal insights
        # In a real scenario, you would implement more sophisticated causal reasoning here
        ndarrays = parameters_to_ndarrays(parameters)
        for i in range(len(ndarrays)):
            # Example: Increase the magnitude of parameters that have a strong causal effect
            ndarrays[i] *= 1.05  # Increase by 5%
        return ndarrays_to_parameters(ndarrays)

def client_fn(context: Context) -> XAIFederatedClient:
    cid = int(context.node_id) % len(trainloaders)
    net = Net().to(DEVICE)

    if cid < len(trainloaders):
        trainloader = trainloaders[cid]
        valloader = valloaders[cid]
    else:
        raise IndexError(f"Client ID {cid} exceeds the number of available data loaders.")

    for i, (images, labels) in enumerate(trainloader):
        print(f"Client {cid} - Batch {i}: images shape = {images.shape}, labels shape = {labels.shape}")

    # Check if the model parameters are correctly initialized
    try:
        initial_parameters = get_parameters(net)
        print(f"Client {cid} parameters initialized successfully.")
    except Exception as e:
        print(f"Failed to get parameters for client {cid}: {e}")
        return None  # or handle it in a way that informs the server

    return XAIFederatedClient(cid, net, trainloader, valloader)
