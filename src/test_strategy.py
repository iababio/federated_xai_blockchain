import pytest
import numpy as np
from src.strategy import XAIFederatedStrategy
from flwr.common import FitRes, Parameters, ndarrays_to_parameters

@pytest.fixture
def strategy():
    """Fixture to create a new instance of XAIFederatedStrategy."""
    return XAIFederatedStrategy()

def test_aggregate_fit(strategy):
    """Test the aggregate_fit method of XAIFederatedStrategy."""
    server_round = 1
    results = [
        (None, FitRes(status="success", parameters=ndarrays_to_parameters([np.array([0.0])]), num_examples=10, metrics={"loss": 0.5, "accuracy": 0.8, "weight_adjustment": 0.1})),
        (None, FitRes(status="success", parameters=ndarrays_to_parameters([np.array([0.0])]), num_examples=10, metrics={"loss": 0.4, "accuracy": 0.85, "weight_adjustment": 0.15})),
    ]
    failures = []

    # Call the aggregate_fit method
    aggregated_parameters, metrics = strategy.aggregate_fit(server_round, results, failures)

    # Check if global history is updated correctly
    assert strategy.global_history["round"][-1] == server_round
    assert len(strategy.global_history["loss"]) == 1
    assert len(strategy.global_history["accuracy"]) == 1
    assert len(strategy.global_history["weight_adjustments"]) == 1

    # Check the values in global history
    assert np.isclose(strategy.global_history["loss"][-1], 0.45)  # Mean of 0.5 and 0.4
    assert np.isclose(strategy.global_history["accuracy"][-1], 0.825)  # Mean of 0.8 and 0.85
    assert np.isclose(strategy.global_history["weight_adjustments"][-1], 0.125)  # Mean of 0.1 and 0.15