from typing import List, Tuple
from flwr.common import Metrics
from flwr.server.strategy import FedAvg, FedProx

import argparse
import flwr as fl

# Define custom metric aggregation function


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    accuracy = 100 * sum(accuracies) / sum(examples)
    return {"accuracy": round(accuracy, 4)}


fraction_fit = 0.8  # Fraction of clients used during training
fraction_evaluate = 0.5  # Fraction of clients used during validation
min_fit_clients = 2  # Minimum number of clients used during training
min_evaluate_clients = 2  # Minimum number of clients used during validation
min_available_clients = 2  # Minimum number of clients available for training
evaluate_metrics_aggregation_fn = (
    weighted_average  # <-- pass the metric aggregation function
)


def startServer(Strat):

    if Strat == "FedAvg":
        strategy = FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    else:
        strategy = FedProx(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
            proximal_mu=0.2,
        )

    config = fl.server.ServerConfig(num_rounds=5)

    fl.server.start_server(
        server_address="0.0.0.0:25565", strategy=strategy, config=config
    )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Configure model aggrigation strategy .")
    parser.add_argument("strategy", choices=["FedAvg", "FedProx"], help="Choose aggregation strategy: Federated Avgeraging[FedAvg] or Federated Optimization[FedProx]")
    args = parser.parse_args()
    
    print(f"Starting server with {args.strategy} strategy")
    startServer(args.strategy)
