import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RESULTS_DIR = "../experiment_results"
PLOTS_DIR = "../plots"


def load_results():
    """
    Loads experiment results from JSON files in the specified RESULTS_DIR.

    This function reads each JSON file, extracts the relevant information about the attack type,
    number of malicious clients, and round-wise accuracy, and compiles them into a Pandas DataFrame.

    Returns:
        pandas.DataFrame: A DataFrame containing the combined data from all the JSON files.
        Columns include 'Round', 'Accuracy', 'Malicious Clients', and 'Attack Type'.
    """
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
    """
    Generates a line plot from the provided DataFrame.

    The function plots the evolution of model accuracy per round, differentiated by the number
    of malicious clients. It saves the plot to the PLOTS_DIR directory.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data to be plotted. It should have
        columns 'Round', 'Accuracy', 'Malicious Clients', and 'Attack Type'.

    Returns:
        Optional[str]: The filename of the saved plot if the DataFrame is not empty;
        otherwise, None.
    """
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
    """
    Main execution block of the script.

    When the script is run directly, it calls `run_plotter` to execute the data loading and plotting process.
    """
    df = load_results()
    plot_filename = plot_results(df)
    print(f"Plot saved as: {os.path.join(PLOTS_DIR, plot_filename)}")


if __name__ == '__main__':
    run_plotter()