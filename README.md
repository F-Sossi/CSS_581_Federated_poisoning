# Federated Learning Attack and Defense Framework

## Overview
This repository hosts the code and resources for our extensive research on the impact of data poisoning attacks on Federated Learning (FL) systems. Our study emphasizes the role of Machine Learning in cybersecurity, with a focus on Federated Learning models trained on decentralized datasets. We explore various poisoning attacks in FL and their implications for model reliability and cyber-security effectiveness.

## Key Features
- **Modular Framework**: A comprehensive and modular setup for simulating a variety of data poisoning attacks in a federated learning environment.
- **Production-Level Testing**: Utilizes the Flower Framework for federated learning, ensuring experiments are conducted in a realistic and production-capable setting.
- **Diverse Attack Simulations**: Includes simulations of random label flipping, constant label flipping, targeted label manipulation, and GAN-based data poisoning.
- **Extensive Experimentation**: Detailed experiments using the CIFAR10 dataset to analyze the effects of different types of attacks on FL models.

## Repository Contents
- **Federated Learning Setup**: Code for setting up the federated learning environment using the Flower framework.
- **Attack Simulations**: Scripts and modules for implementing various poisoning attacks.
- **Model and Data Configuration**: Resources for setting up the Convolutional Neural Network (CNN) and data preprocessing.
- **GAN Implementation**: Generative Adversarial Network setup and integration with the FL experiments.
- **Experimental Results**: Visualization and analysis scripts for interpreting the outcomes of the experiments.

## Research Findings
Our research highlights the significant effects of data poisoning attacks in FL, especially the role of malicious client proportion in attack effectiveness. We demonstrate the varying impacts of different attack types on model performance, emphasizing the need for robust defenses in FL systems.

## Future Directions
The repository also outlines potential future work, including the exploration of more sophisticated client models, refined GAN strategies, broader attack spectrums, and real-world deployment scenarios.

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

From the command line:

python orchestrator.py attacktype

attacktype is a command line argument (with no argument: default is random_flip)

#### Supported Attack Types so Far:

- gan_attack (must run gan.py first to generate fake data batch size deterimines the number of fake images)

- random_flip

- constant_flip_X

(X is an int which will be used to shift all labels down by that amount, for example 'constant_flip_3' will change 
label 3 to 0, 4 to 1, 2 to 9, 1 to 8, 0 to 7 and etc.)

- targeted_TX_TY

(X is the original label, and Y is the label it will be changed to)
This leaves the rest of the labels unchanged


#### Examples
```shell


python orchestrator.py constant_flip5

python orchestrator.py targeted_T1T0

python orchestrator.py targeted_T8T4

```


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

