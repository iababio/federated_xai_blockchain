import flwr as fl
import numpy as np
from flwr.common import Parameters, Scalar, Status, Code  # Ensure Parameters is imported
from typing import List, Tuple, Optional, Dict, Union
from flwr.common import FitRes

class XAIFederatedStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__()
        self.global_history = {
            'round': [],
            'loss': [],
            'accuracy': [],
            'weight_adjustments': []
        }

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if results:
            round_metrics = {
                'loss': np.mean([r.metrics['loss'] for _, r in results]),
                'accuracy': np.mean([r.metrics['accuracy'] for _, r in results]),
                'weight_adjustments': np.mean([r.metrics['weight_adjustment'] for _, r in results])
            }

            self.global_history['round'].append(server_round)
            self.global_history['loss'].append(round_metrics['loss'])
            self.global_history['accuracy'].append(round_metrics['accuracy'])
            self.global_history['weight_adjustments'].append(round_metrics['weight_adjustments'])

        return aggregated_parameters, metrics

    def smart_contract(self):
        return self.global_history['weight_adjustments']

    def optimize_federation(self, parameters: Parameters) -> Parameters:
        ndarrays = parameters_to_ndarrays(parameters)
        for i in range(len(ndarrays)):
            ndarrays[i] *= 1.05  # Increase by 5%
        return ndarrays_to_parameters(ndarrays)