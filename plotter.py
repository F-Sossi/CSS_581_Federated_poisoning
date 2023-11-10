import os
import json
import matplotlib.pyplot as plt

RESULTS_DIR = "experiment_results"


def load_results():
    results = {}
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith(".json"):
            num_malicious = int(filename.split("_")[1])
            with open(os.path.join(RESULTS_DIR, filename), 'r') as f:
                results[num_malicious] = json.load(f)
    return results


def plot_results(results):
    num_malicious_clients = sorted(results.keys())
    accuracies = [sum([acc["accuracy"] for acc in results[num]]) / len(results[num]) for num in num_malicious_clients]

    plt.figure(figsize=(6, 5))

    # Plotting accuracy
    plt.plot(num_malicious_clients, accuracies, marker='o')
    plt.title('Average Accuracy vs. Number of Malicious Clients')
    plt.xlabel('Number of Malicious Clients')
    plt.ylabel('Average Accuracy')

    plt.tight_layout()
    plt.show()


def main():
    results = load_results()
    plot_results(results)


if __name__ == "__main__":
    main()
