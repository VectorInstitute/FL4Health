# Client Level Differential Privacy Federated Learning Example

This example shows how to implement Differential Privacy into the Federated Learning framework. In this case we focus on *client level* privacy which is a more substantial version of instance level DP, where the participation of an entire client's set of data is protected from training dataset membership inference. This example uses the FedAvgM implementation with weighted averaging suggested in Learning Differentially Private Recurrent Language Models along with the adaptive clipping scheme proposed in Differentially Private Learning with Adaptive Clipping. The example uses an accountant specifically tailered to this approach. The clients are Poisson sampled by default.

The example involves collaboratively learning a logistic regression model accross multiple hospitals to classify breast cancer given 31 features. The dataset is sourced from [kaggle](https://www.kaggle.com/competitions/breast-cancer-classification/overview). A processed federated version of the dataset is available in the repository.

# Running the Example
In order to run the example, first ensure you have the virtual env of your choice activated and run
```
pip install --upgrade pip
pip install -r requirements.txt
```
to install all of the dependencies for this project.

## Starting Server

The next step is to start the server by running
```
python -m examples.dp_fed_examples.client_level_dp_weighted.server --config_path examples/dp_fed_examples/client_level_dp_weighted/config.yaml
```

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the clients expected by the server. This is done by simply running (remembering to activate your environment)
```
python -m examples.dp_fed_examples.client_level_dp_weighted.client --dataset_path examples/datasets/breast_cancer_data/hospital_#.csv
```
After the minimum number of clients have been started federated learning should commence.
