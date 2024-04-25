import matplotlib.pyplot as plt
import os, json,itertools, statistics


federated_tasks = ['fed_heart_disease', 'fed_isic2019', 'fed_ixi']
dp_types = ['local', 'central', 'distributed']
hyper_parameter_names = ['noise']
hyper_parameter_values = {
    'fed_heart_disease': [1, 3, 5],
    'fed_isic2019': [2, 4, 6], 
    'fed_ixi': [1, 4, 9],
}

experiments = list(itertools.product(federated_tasks, dp_types, hyper_parameter_names))

experiments_full = []
for tuple in experiments:
    values = hyper_parameter_values[tuple[0]]
print(experiments)






# PATH='logs/log/fed_heart_disease_local/hp_sweep_results/gaussian_noise_variance_0.001/Run1/metrics/server_metrics.json'

# with open(PATH) as file:
#     records = json.load(file)
#     print(type(records['train_meter_FedHeartDisease_accuracy']))
    