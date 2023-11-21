
### Running the Project
```angular2html
# from src/ directory
python orchestrator.py attack_type

# attacks types:
# constant_flip
# random_flip
# targeted_flip

# example:
python orchestrator.py constant_flip
```

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
├── fakeData
│   ├── fake_samples_epoch_014.png
│   └── real_samples.png
├── log_metrics
│   ├── targetedT1T2Round0_ID0_.csv
│   ├── targetedT1T2Round0_ID1_.csv
│   ├── targetedT1T2Round0_ID2_.csv
│   ├── targetedT1T2Round0_ID3_.csv
│   ├── targetedT1T2Round0_ID4_.csv
│   ├── targetedT1T2Round1_ID0_.csv
│   ├── targetedT1T2Round1_ID1_.csv
│   ├── targetedT1T2Round1_ID2_.csv
│   ├── targetedT1T2Round1_ID3_.csv
│   ├── targetedT1T2Round1_ID4_.csv
│   ├── targetedT1T2Round2_ID0_.csv
│   ├── targetedT1T2Round2_ID1_.csv
│   ├── targetedT1T2Round2_ID2_.csv
│   ├── targetedT1T2Round2_ID3_.csv
│   └── targetedT1T2Round2_ID4_.csv
├── README.md
├── requirements.txt
├── src
│   ├── client.py
│   ├── gan2.py
│   ├── gan.py
│   ├── orchestrator.py
│   ├── plotter.py
│   └── server.py
├── src\log_metrics\random_flipRound0_ID0_.csv
├── src\log_metrics\random_flipRound0_ID1_.csv
├── src\log_metrics\random_flipRound0_ID2_.csv
├── src\log_metrics\random_flipRound0_ID3_.csv
├── src\log_metrics\random_flipRound0_ID4_.csv
├── src\log_metrics\random_flipRound0_ID5_.csv
├── src\log_metrics\random_flipRound0_ID6_.csv
├── src\log_metrics\random_flipRound0_ID7_.csv
├── src\log_metrics\random_flipRound0_ID8_.csv
├── src\log_metrics\random_flipRound0_ID9_.csv
└── weights

```

