import subprocess
import threading
import os
import sys
import time

print('Running Orchestrator (testJGN)')
# Parameters

NUM_TOTAL_CLIENTS = 3
MAX_MALICIOUS_CLIENTS = 2
NUM_ROUNDS = 3
RESULTS_DIR = "../experiment_results"

#An experiment ID
EXP_ID = 'N_total' + str(NUM_TOTAL_CLIENTS)
EXP_ID += '_Max_mal' + str(MAX_MALICIOUS_CLIENTS)
EXP_ID += 'N_rounds' + str(NUM_ROUNDS)

cwd = os.getcwd()
path = cwd.replace('\\src','')
path += '\\log_metrics\\' + EXP_ID

try:
    os.makedirs(path)
except FileExistsError:
    # directory already exists
    pass

"""
ATTACK TYPES:
random_flip
constant_flip_X # substitute X with the offset (if 0 or 10 labels will be unchanged)
targeted_TXTY  # substitute X with the label to be changed, Y the label it is changed to
gan_attack # Use GAN fake data
"""


# Function to start the Flower server
def start_server(num_rounds, output_file, attack):
    cmd = ["python", "server.py", "--rounds", str(num_rounds), "--output", output_file, '--attack', attack]
    subprocess.run(cmd)


# Function to start a Flower client
def start_client(is_malicious=False, attack='none', client_id=0, num_mal=0, exp_id='undefined'):
    env = os.environ.copy()
    # print("Starting client. Malicious:", is_malicious)
    env["IS_MALICIOUS"] = "1" if is_malicious else "0"
    env["ATTACK"] = str(attack)
    env["CLIENT_ID"] = str(client_id)
    env["NUM_MAL"] = str(num_mal)
    env['EXP_ID'] = exp_id
    cmd = ["python", "client.py"]
    subprocess.Popen(cmd, env=env)
    # print("Client finished.")


# Main function to orchestrate the server and clients
def main():
    if len(sys.argv) > 1:
        attack = sys.argv[1]
    else:
        attack = 'random_flip'
        print('default attack type chosen:', attack)
    print('orchestrator main, attack type:', attack)

    for num_malicious in range(MAX_MALICIOUS_CLIENTS + 1):
        # For testing
        # for num_malicious in range(1,2):
        print(f"Running experiment with {num_malicious} malicious clients, attack type:", attack)

        # Start the server in a separate thread
        output_file = os.path.join(RESULTS_DIR, f"{attack}results_{num_malicious}_malicious.json")
        server_thread = threading.Thread(target=start_server, args=(NUM_ROUNDS, output_file, attack))
        server_thread.start()
        time.sleep(10)  # Wait for the server to initialize

        # Start clients
        client_threads = []
        for i in range(NUM_TOTAL_CLIENTS):
            client_id = i
            is_malicious = i < num_malicious
            attack_type = attack
            print('creating client', 'malicious:', is_malicious, ', attack_type:', attack_type)
            client_thread = threading.Thread(target=start_client,
                                             args=(is_malicious, attack_type, client_id, num_malicious, EXP_ID))
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
