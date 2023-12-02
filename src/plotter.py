import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RESULTS_DIR = "../experiment_results"
PLOTS_DIR = "../plots"


def load_results():
    all_data = []
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith(".json"):
            try:
                # Splitting filename and extracting relevant parts
                parts = filename.split("results_")
                attack_type = parts[0]
                num_malicious = int(parts[1].split("_malicious")[0])
            except (IndexError, ValueError):
                print(f"Unable to extract details from the filename: {filename}, skipping this file.")
                continue

            with open(os.path.join(RESULTS_DIR, filename), 'r') as f:
                accuracies = json.load(f)
                if not accuracies:
                    print(f"The file {filename} does not contain any data, skipping this file.")
                    continue
                for round_num, acc in enumerate(accuracies):
                    if "accuracy" not in acc:
                        print(f"The file {filename} does not contain 'accuracy' data, skipping this data.")
                        continue
                    all_data.append(
                        {"Round": round_num, "Accuracy": acc["accuracy"],
                         "Malicious Clients": num_malicious, "Attack Type": attack_type})
    return pd.DataFrame(all_data)


def plot_results(df):
    if df.empty:
        print("No valid data to plot.")
        return

    sns.set_style("whitegrid")
    sns.set_palette("deep")

    plt.figure(figsize=(10, 6))
    sns.lineplot(x="Round", y="Accuracy", hue="Malicious Clients", data=df, markers=True)

    # Assuming all rows have the same attack type
    attack_type = df['Attack Type'].iloc[0]
    title = f'Model Accuracy Evolution per Round (Attack: {attack_type})'

    plt.title(title)
    plt.xlabel('Round')
    plt.ylabel('Accuracy')

    plt.tight_layout()

    # Save the plot
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    plot_filename = f"model_accuracy_{attack_type}.png"
    plt.savefig(os.path.join(PLOTS_DIR, plot_filename))
    plt.show()

    return plot_filename  # Return the filename for reference


def run_plotter():
    df = load_results()
    plot_filename = plot_results(df)
    print(f"Plot saved as: {os.path.join(PLOTS_DIR, plot_filename)}")


