"""
This module implements a custom federated learning strategy using Flower's FedAvg.
It tracks global history metrics, emulates smart contracts, and optimizes parameters.
"""

from typing import List, Tuple, Optional, Dict, Union
import flwr as fl
import numpy as np
from flwr.common import (
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.common import FitRes

from src.blockchain import W3, dataset_contract


class XAIFederatedStrategy(fl.server.strategy.FedAvg):
    """
    Custom federated learning strategy with tracking of global history metrics,
    smart contract emulation, and parameter optimization.
    """

    def __init__(self):
        """
        Initialize the strategy with an empty global history dictionary
        to store training metrics across rounds.
        """
        super().__init__()
        self.global_history = {
            "round": [],
            "loss": [],
            "accuracy": [],
            "weight_adjustments": [],
        }
        self.contract = dataset_contract  # Smart contract instance
        self.web3 = W3  # Web3 instance connected to the blockchain

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[
            Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]
        ],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregates client model updates after each federated round, calculates
        round metrics, and updates the global history.

        Args:
            server_round (int): The current round of federated learning.
            results (List[Tuple[ClientProxy, FitRes]]): Results from each client.
            failures (List[Union[Tuple[ClientProxy, FitRes], BaseException]]): 
            Failed client updates.

        Returns:
            Tuple[Optional[Parameters], Dict[str, Scalar]]: Aggregated parameters and metrics.
        """
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if results:
            round_metrics = {
                "loss": np.mean([r.metrics["loss"] for _, r in results]),
                "accuracy": np.mean([r.metrics["accuracy"] for _, r in results]),
                "weight_adjustments": np.mean(
                    [r.metrics["weight_adjustment"] for _, r in results]
                ),
            }

            self.global_history["round"].append(server_round)
            self.global_history["loss"].append(round_metrics["loss"])
            self.global_history["accuracy"].append(round_metrics["accuracy"])
            self.global_history["weight_adjustments"].append(
                round_metrics["weight_adjustments"]
            )

        return aggregated_parameters, metrics

    def save_weights_to_blockchain(self, parameters):
        # Convert parameters to a format suitable for blockchain storage
        weights_hash = self.hash_parameters(parameters)

        # Store the hashed weights reference on the blockchain
        tx_hash = self.contract.functions.storeWeights(
            weights_hash  # Store the hashed reference to weights
        ).transact({'from': self.web3.eth.accounts[0]})

        # Wait for the transaction to be mined
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"Stored weights on blockchain, TxHash = {tx_hash.hex()}")

    def hash_parameters(self, parameters) -> str:
        # Flatten and hash parameters for storage reference
        flattened_weights = [param.flatten() for param in parameters]
        weights_array = np.concatenate(flattened_weights)
        return self.web3.toHex(self.web3.soliditySha3(['bytes32'], [weights_array.tobytes()]))

    def optimize_federation(self, parameters: Parameters) -> Parameters:
        """
        Scales parameters by a factor of 1.05 (5% increase) to optimize federation.

        Args:
            parameters (Parameters): Model parameters.

        Returns:
            Parameters: Optimized parameters after scaling.
        """
        ndarrays = parameters_to_ndarrays(parameters)
        for i, param in enumerate(ndarrays):
            ndarrays[i] = param * 1.05  # Increase by 5%
        return ndarrays_to_parameters(ndarrays)
