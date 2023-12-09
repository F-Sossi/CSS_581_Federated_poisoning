import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RESULTS_DIR = "../experiment_results_ext"
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
                        {"Round": round_num,
                         "accuracy": acc['accuracy'],
                         "adversarial_accuracy": acc["adversarial_accuracy"],
                         "target_precision": acc['target_precision'],
                         "target_recall": acc['target_recall'],
                         "new_precision": acc['new_precision'],
                         "new_recall": acc['new_recall'],
                         "adversarial_precision_wgt": acc['adversarial_precision_wgt'],
                         "adversarial_recall_wgt": acc['adversarial_recall_wgt'],
                         "adversarial_precision_wgnl": acc['adversarial_precision_wgnl'],
                         "adversarial_recall_wgnl": acc['adversarial_recall_wgnl'],
                         "Malicious Clients": num_malicious, "Attack Type": attack_type})
    return pd.DataFrame(all_data)


def plot_results(df, metric):
    if df.empty:
        print("No valid data to plot.")
        return

    sns.set_style("whitegrid")
    sns.set_palette("deep")

    plt.figure(figsize=(10, 6))
    sns.lineplot(x="Round", y=metric, hue="Malicious Clients", data=df, markers=True)

    # Assuming all rows have the same attack type
    attack_type = df['Attack Type'].iloc[0]
    title = f'TargetFlip {metric} Evolution per Round (Attack: {attack_type})'

    plt.title(title)
    plt.xlabel('Round')
    plt.ylabel(metric)

    plt.tight_layout()

    # Save the plot
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    plot_filename = f"targetflip_{metric}_{attack_type}.png"
    plt.savefig(os.path.join(PLOTS_DIR, plot_filename))
    #plt.show()

    return plot_filename  # Return the filename for reference

def plot_results_MaxRound(df, metrics):
    if df.empty:
        print("No valid data to plot.")
        return
    # Assuming all rows have the same attack type
    attack_type = df['Attack Type'].iloc[0]
    MaxRound=0
    rounds = list(df['Round'].values)
    MaxRound = max(rounds)
    MinRound = min(rounds)
    print('min, max:', MinRound, MaxRound)

    df = df[df['Round'] == MaxRound]
    df = df[metrics + ['Malicious Clients']]
    sns.set_style("whitegrid")
    sns.set_palette("deep")

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Malicious Clients', y='value', hue='variable',
                 data=pd.melt(df, ['Malicious Clients']))

    title = f'TargetFlip Metrics Evolution per Round (Attack: {attack_type})'

    plt.title(title)
    plt.xlabel('Malicious Clients')
    plt.ylabel('metric value')

    plt.tight_layout()

    # Save the plot
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    plot_filename = f"Maxroundtargetflip_metrics_{attack_type}.png"
    plt.savefig(os.path.join(PLOTS_DIR, plot_filename))
    #plt.show()

    return plot_filename  # Return the filename for reference


def run_plotter():
    df = load_results()
    metrics = ['accuracy', 'adversarial_accuracy','target_precision','target_recall','new_precision',
               'new_recall', 'adversarial_precision_wgt', 'adversarial_recall_wgt',
               'adversarial_precision_wgnl', 'adversarial_recall_wgnl']

    plot_filename = plot_results_MaxRound(df, metrics)
    print(f"Plot saved as: {os.path.join(PLOTS_DIR, plot_filename)}")

    for metric in metrics:
        plot_filename = plot_results(df, metric)
        print(f"Plot saved as: {os.path.join(PLOTS_DIR, plot_filename)}")

if __name__ == "__main__":
    run_plotter()
