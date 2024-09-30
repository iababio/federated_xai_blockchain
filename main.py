client_ids = [str(i) for i in range(num_clients)]

# Modify the FedSparse strategy to include the evaluation function
strategy = FedSparse()

# Continue with the simulation setup
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=num_clients,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
    client_resources={"num_cpus": 1, "num_gpus": 0}
)