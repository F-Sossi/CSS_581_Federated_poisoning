import os
import json
import matplotlib.pyplot as plt

RESULTS_DIR = "experiment_results"

<<<<<<< HEAD

=======
>>>>>>> 8872afb (init)
def load_results():
    results = {}
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith(".json"):
            num_malicious = int(filename.split("_")[1])
            with open(os.path.join(RESULTS_DIR, filename), 'r') as f:
                results[num_malicious] = json.load(f)
    return results

<<<<<<< HEAD

=======
>>>>>>> 8872afb (init)
def plot_results(results):
    num_malicious_clients = sorted(results.keys())
    accuracies = [results[num]["accuracy"] for num in num_malicious_clients]
    losses = [results[num]["loss"] for num in num_malicious_clients]

    plt.figure(figsize=(12, 5))

    # Plotting accuracy
    plt.subplot(1, 2, 1)
    plt.plot(num_malicious_clients, accuracies, marker='o')
    plt.title('Accuracy vs. Number of Malicious Clients')
    plt.xlabel('Number of Malicious Clients')
    plt.ylabel('Accuracy')

    # Plotting loss
    plt.subplot(1, 2, 2)
    plt.plot(num_malicious_clients, losses, marker='o', color='red')
    plt.title('Loss vs. Number of Malicious Clients')
    plt.xlabel('Number of Malicious Clients')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()

<<<<<<< HEAD

=======
>>>>>>> 8872afb (init)
def main():
    results = load_results()
    plot_results(results)

<<<<<<< HEAD

=======
>>>>>>> 8872afb (init)
if __name__ == "__main__":
    main()
