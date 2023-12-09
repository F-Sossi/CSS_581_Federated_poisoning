from typing import List, Tuple, Optional, Dict
import flwr as fl
from flwr.common import Parameters, Metrics, Scalar, EvaluateRes, FitRes
import argparse
import os
import json
import threading


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Accumulate weighted sums and total examples
    weighted_sum = sum(num_examples * m["accuracy"] for num_examples, m in metrics if "accuracy" in m)
    total_examples = sum(num_examples for num_examples, _ in metrics)

    # Calculate weighted average
    return {"accuracy": weighted_sum / total_examples if total_examples > 0 else 0}


class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_log = []
        self.extended_metrics_log = []

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

        aggregated_loss2, aggregated_metrics2 = super().aggregate_evaluate(server_round, results, failures)

        # Add custom logic to aggregate metrics such as accuracy
        # Extended Metrics
        if results:

            #Adversrial Accuracy
            weighted_sum = sum(
                res.metrics["adversarial_accuracy"] * res.num_examples for _, res in results if "adversarial_accuracy" in res.metrics)
            total_examples = sum(res.num_examples for _, res in results)
            aggregated_metrics2["adversarial_accuracy"] = weighted_sum / total_examples if total_examples > 0 else 0

            #target Precision
            weighted_sum = sum(
                res.metrics["target_precision"] * res.num_examples for _, res in results if
                "target_precision" in res.metrics)
            total_examples = sum(res.num_examples for _, res in results)
            aggregated_metrics2["target_precision"] = weighted_sum / total_examples if total_examples > 0 else 0

            # target Recall
            weighted_sum = sum(
                res.metrics["target_recall"] * res.num_examples for _, res in results if
                "target_recall" in res.metrics)
            total_examples = sum(res.num_examples for _, res in results)
            aggregated_metrics2["target_recall"] = weighted_sum / total_examples if total_examples > 0 else 0

            # new Precision
            weighted_sum = sum(
                res.metrics["new_precision"] * res.num_examples for _, res in results if
                "new_precision" in res.metrics)
            total_examples = sum(res.num_examples for _, res in results)
            aggregated_metrics2["new_precision"] = weighted_sum / total_examples if total_examples > 0 else 0

            # new Recall
            weighted_sum = sum(
                res.metrics["new_recall"] * res.num_examples for _, res in results if
                "new_recall" in res.metrics)
            total_examples = sum(res.num_examples for _, res in results)
            aggregated_metrics2["new_recall"] = weighted_sum / total_examples if total_examples > 0 else 0

            # adversarial precision with regards to target
            weighted_sum = sum(
                res.metrics["adversarial_precision_wgt"] * res.num_examples for _, res in results if
                "adversarial_precision_wgt" in res.metrics)
            total_examples = sum(res.num_examples for _, res in results)
            aggregated_metrics2["adversarial_precision_wgt"] = weighted_sum / total_examples if total_examples > 0 else 0

            # adversarial Recall with regards to target
            weighted_sum = sum(
                res.metrics["adversarial_recall_wgt"] * res.num_examples for _, res in results if
                "adversarial_recall_wgt" in res.metrics)
            total_examples = sum(res.num_examples for _, res in results)
            aggregated_metrics2["adversarial_recall_wgt"] = weighted_sum / total_examples if total_examples > 0 else 0

            # adversarial precision with regards to new label
            weighted_sum = sum(
                res.metrics["adversarial_precision_wgnl"] * res.num_examples for _, res in results if
                "adversarial_precision_wgnl" in res.metrics)
            total_examples = sum(res.num_examples for _, res in results)
            aggregated_metrics2["adversarial_precision_wgnl"] = weighted_sum / total_examples if total_examples > 0 else 0

            # adversarial Recall with regards to new label
            weighted_sum = sum(
                res.metrics["adversarial_recall_wgnl"] * res.num_examples for _, res in results if
                "adversarial_recall_wgnl" in res.metrics)
            total_examples = sum(res.num_examples for _, res in results)
            aggregated_metrics2["adversarial_recall_wgnl"] = weighted_sum / total_examples if total_examples > 0 else 0


        self.extended_metrics_log.append(aggregated_metrics2)

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


filename = args.output
filename = filename.replace('.json', '_ext.json')
filename = filename.replace('experiment_results', 'experiment_results_ext')
with open(filename, "w+") as f:
    json.dump(strategy.extended_metrics_log, f)
