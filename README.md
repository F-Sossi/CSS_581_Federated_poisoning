
### Installing Dependencies

Project dependencies (such as `torch` and `flwr`) are defined in `requirements.txt`. 
#### Pycharm

```shell
Open folder pycharm will run the requirments.txt
```

#### Pip

Write the command below in your terminal to install the dependencies according to the configuration file requirements.txt.

```shell
pip install -r requirements.txt
```

#### Run Federated Learning with PyTorch and Flower

run orchestrator.py

From the command line:

python orchestrator.py attacktype

attacktype is a command line argument (with no argument: default is random_flip)

Supported Attack Types so Far:

random_flip

constant_flip_X

(X is an int which will be used to shift all labels down by that amount, for example 'constant_flip_3' will change 
label 3 to 0, 4 to 1, 2 to 9, 1 to 8, 0 to 7 and etc.)

targeted_TX_TY

(X is the original label, and Y is the label it will be changed to)
This leaves the rest of the labels unchanged

Examples

python orchestrator.py constant_flip5

python orchestrator.py targeted_T1T0

python orchestrator.py targeted_T8T4


### File Structure

```shell
├── data
│   ├── cifar-10-batches-py
│   │   ├── batches.meta
│   │   ├── data_batch_1
│   │   ├── data_batch_2
│   │   ├── data_batch_3
│   │   ├── data_batch_4
│   │   ├── data_batch_5
│   │   ├── readme.html
│   │   └── test_batch
│   └── cifar-10-python.tar.gz
├── experiment_results
│   ├── results_0_malicious.json
│   ├── results_10_malicious.json
│   ├── results_1_malicious.json
│   ├── results_2_malicious.json
│   ├── results_3_malicious.json
│   ├── results_4_malicious.json
│   ├── results_5_malicious.json
│   ├── results_6_malicious.json
│   ├── results_7_malicious.json
│   ├── results_8_malicious.json
│   └── results_9_malicious.json
├── pyproject.toml
├── README.md
├── requirements.txt
├── run.sh
└── src
    ├── client.py
    ├── orchestrator.py
    ├── plotter.py
    └── server.py

5 directories, 28 files
```

