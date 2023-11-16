import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RESULTS_DIR = "experiment_results"


def load_results():
    all_data = []
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith(".json"):
            num_malicious = int(filename.split("_")[1])
            with open(os.path.join(RESULTS_DIR, filename), 'r') as f:
                accuracies = json.load(f)
                for round_num, acc in enumerate(accuracies):
                    all_data.append(
                        {"Round": round_num, "Accuracy": acc["accuracy"], "Malicious Clients": num_malicious})
    return pd.DataFrame(all_data)


def plot_results(df):
    sns.set_style("whitegrid")  # Example styles: "darkgrid", "whitegrid", "dark", "white", and "ticks"
    # sns.set_context("talk")  # Other contexts include "paper", "notebook", "talk", and "poster"
    sns.set_palette("deep")  # Example palettes: "pastel", "muted", "bright", "deep", "colorblind"

    plt.figure(figsize=(10, 6))
    sns.lineplot(x="Round", y="Accuracy", hue="Malicious Clients", data=df, marker='o')

    plt.title('Accuracy Evolution per Round by Number of Malicious Clients')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.show()


def main():
    df = load_results()
    plot_results(df)


if __name__ == "__main__":
    main()
