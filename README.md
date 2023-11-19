

```shell
-- pyproject.toml
-- requirements.txt
-- client.py
-- server.py
-- README.md
```

### Installing Dependencies

Project dependencies (such as `torch` and `flwr`) are defined in `pyproject.toml` and `requirements.txt`. 
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

