from typing import List, Tuple
import flwr as fl
from flwr.common import Metrics, Scalar
import argparse
import json


# Define metric aggregation functions
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Accumulate weighted sums and total examples
    weighted_sum = sum(num_examples * m["accuracy"] for num_examples, m in metrics if "accuracy" in m)
    total_examples = sum(num_examples for num_examples, _ in metrics)

    # Calculate weighted average
    return {"accuracy": weighted_sum / total_examples if total_examples > 0 else 0}


# Parse command line arguments
parser = argparse.ArgumentParser(description="Flower server with custom strategy")
parser.add_argument("--rounds", type=int, default=3, help="Number of federated learning rounds")
parser.add_argument("--output", type=str, default="results.json", help="Output file for results")
args = parser.parse_args()

# Define strategy
strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average
)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=args.rounds),
    strategy=strategy
)

# Save results to a file
with open(args.output, "w") as f:
    json.dump(strategy.metrics, f)
