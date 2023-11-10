import subprocess
import threading
import os
import time

# Parameters
NUM_TOTAL_CLIENTS = 5
MAX_MALICIOUS_CLIENTS = 2
NUM_ROUNDS = 5
RESULTS_DIR = "experiment_results"


# Function to start the Flower server
def start_server(num_rounds, output_file):
    cmd = ["python", "server.py", "--rounds", str(num_rounds), "--output", output_file]
    subprocess.run(cmd)


# Function to start a Flower client
def start_client(is_malicious=False):
    env = os.environ.copy()
    print("Starting client. Malicious:", is_malicious)
    env["IS_MALICIOUS"] = "1" if is_malicious else "0"
    cmd = ["python", "client.py"]
    subprocess.run(cmd, env=env)
    print("Client finished.")


# Main function to orchestrate the server and clients
def main():
    for num_malicious in range(MAX_MALICIOUS_CLIENTS + 1):
        print(f"Running experiment with {num_malicious} malicious clients")

        # Start the server in a separate thread
        output_file = os.path.join(RESULTS_DIR, f"results_{num_malicious}_malicious.json")
        server_thread = threading.Thread(target=start_server, args=(NUM_ROUNDS, output_file))
        server_thread.start()
        time.sleep(5)  # Wait for the server to initialize

        # Start clients
        client_threads = []
        for i in range(NUM_TOTAL_CLIENTS):
            is_malicious = i < num_malicious
            client_thread = threading.Thread(target=start_client, args=(is_malicious,))
            client_threads.append(client_thread)
            client_thread.start()

        # Wait for all clients to complete
        for thread in client_threads:
            thread.join()

        # Wait for the server to complete
        server_thread.join()
        print(f"Experiment with {num_malicious} malicious clients completed")


if __name__ == "__main__":
    main()


