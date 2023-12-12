from typing import List, Tuple, Optional, Dict
import flwr as fl
from flwr.common import Parameters, Metrics, Scalar, EvaluateRes, FitRes
import argparse
import json
import threading


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Computes the weighted average of given metrics.

    Parameters:
        metrics (List[Tuple[int, Metrics]]): A list of tuples, where each tuple contains
        the number of examples and the metrics dictionary for a client.

    Returns:
        Metrics: A dictionary containing the weighted average of the metrics.
    """
    weighted_sum = sum(num_examples * m["accuracy"] for num_examples, m in metrics if "accuracy" in m)
    total_examples = sum(num_examples for num_examples, _ in metrics)

    # Calculate weighted average
    return {"accuracy": weighted_sum / total_examples if total_examples > 0 else 0}


class CustomFedAvg(fl.server.strategy.FedAvg):
    """
    Custom federated averaging strategy for Flower federated learning server.

    Inherits from fl.server.strategy.FedAvg.

    Attributes:
        metrics_log (list): Logs aggregated metrics for each round.

    Methods:
        aggregate_evaluate(server_round, results, failures): Custom aggregation of evaluation
        results including additional logic for metrics aggregation.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the CustomFedAvg strategy with given arguments.

        Args and kwargs are passed to the superclass FedAvg's __init__ method.
        """
        super().__init__(*args, **kwargs)
        self.metrics_log = []

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_manager.ClientProxy, EvaluateRes]],
            failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        print(f"\n Round {server_round}: Aggregating evaluation results")

        # Add custom logic to aggregate metrics such as accuracy
        if results:
            weighted_sum = sum(
                res.metrics["accuracy"] * res.num_examples for _, res in results if "accuracy" in res.metrics)
            total_examples = sum(res.num_examples for _, res in results)
            aggregated_metrics["accuracy"] = weighted_sum / total_examples if total_examples > 0 else 0

        # Log metrics for each round
        self.metrics_log.append(aggregated_metrics)

        return aggregated_loss, aggregated_metrics


# Parse command line arguments
parser = argparse.ArgumentParser(description="Flower server with custom strategy")
parser.add_argument("--rounds", type=int, default=3, help="Number of federated learning rounds")
parser.add_argument("--output", type=str, default="results.json", help="Output file for results")
parser.add_argument("--attack", type=str, default="random_flip", help="Output file for results")

args = parser.parse_args()

# Define strategy with custom class
strategy = CustomFedAvg(
    evaluate_metrics_aggregation_fn=weighted_average
)


def run_server():
    """
    Starts the Flower federated learning server with the defined strategy.

    The server configuration and strategy are defined globally. This function initializes
    the server and begins the federated learning process.
    """
    fl.server.start_server(
        server_address="localhost:8081",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy
    )


# Start the server in a separate thread
server_thread = threading.Thread(target=run_server)
server_thread.start()
server_thread.join()  # Wait for the server thread to finish

# Save results to a file after the server has finished
with open(args.output, "w+") as f:
    json.dump(strategy.metrics_log, f)
