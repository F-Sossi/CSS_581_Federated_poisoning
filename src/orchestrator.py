import subprocess
import threading
import os
import sys
import time

print('Running Orchestrator (testJGN)')
# Parameters
NUM_TOTAL_CLIENTS = 5
MAX_MALICIOUS_CLIENTS = 2
NUM_ROUNDS = 5
RESULTS_DIR = "../experiment_results"

"""
random_flip
constant_flip_X # substitute X with the offset (if 0 or 10 labels will be unchanged)
targeted_TXTY  # substitute X with the label to be changed, Y the label it is changed to
"""
ATTACK='targeted_T1T2'


# Function to start the Flower server
def start_server(num_rounds, output_file, ATTACK):
    cmd = ["python", "server.py", "--rounds", str(num_rounds), "--output", output_file, '--attack', ATTACK]
    subprocess.run(cmd)


# Function to start a Flower client
def start_client(is_malicious=False, attack='none',client_id=0,round_number=0):
    env = os.environ.copy()
    # print("Starting client. Malicious:", is_malicious)
    env["IS_MALICIOUS"] = "1" if is_malicious else "0"
    env["ATTACK"]=str(attack)
    env["CLIENT_ID"]=str(client_id)
    env["ROUND"]=str(round_number)
    cmd = ["python", "client.py"]
    subprocess.Popen(cmd, env=env)
    # print("Client finished.")


# Main function to orchestrate the server and clients
def main():
    ATTACK=sys.argv[1]
    print(ATTACK)

    for num_malicious in range(MAX_MALICIOUS_CLIENTS + 1):
    # For testing
    #for num_malicious in range(1,2):
        print(f"Running experiment with {num_malicious} malicious clients, attack type:", ATTACK)

        # Start the server in a separate thread
        output_file = os.path.join(RESULTS_DIR, f"{ATTACK}results_{num_malicious}_malicious.json")
        server_thread = threading.Thread(target=start_server, args=(NUM_ROUNDS, output_file, ATTACK))
        server_thread.start()
        time.sleep(5)  # Wait for the server to initialize

        # Start clients
        client_threads = []
        for i in range(NUM_TOTAL_CLIENTS):
            client_id=i
            round_number=num_malicious
            is_malicious = i < num_malicious
            if not is_malicious:
                attack_type='none'
            else:
                attack_type=ATTACK
            print('creating client', 'malicious:',is_malicious, ', attack_type:', attack_type)
            client_thread = threading.Thread(target=start_client, args=(is_malicious,ATTACK,client_id,round_number,))
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


