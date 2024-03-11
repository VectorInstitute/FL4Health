import matplotlib.pyplot as plt
import os
import json
import statistics
from fl4health.privacy.distributed_discrete_gaussian_accountant import DistributedDiscreteGaussianAccountant

heart = {'name': 'fed_heart_disease',
         'metric': 'train_meter_FedHeartDisease_accuracy',
         'loss': 'loss',
         'error': 'round vs l_inf_error',
         'dimension': 14,
         'num_clients': 4,}

isic = {'name': 'fed_isic2019',
         'metric': 'train_meter_FedIsic2019_balanced_accuracy',
         'loss': 'loss',
         'error': 'round vs l_inf_error',
        'dimension': 4017796,
         'num_clients': 6,
         }

ixi = {'name': 'fed_ixi',
         'metric': 'train_meter_FedIXI_dice',
         'loss': 'loss',
         'error': 'round vs l_inf_error',
        'dimension': 492312,
         'num_clients': 3,}


for exp in [heart, isic, ixi]:
    ############ change these ##########
    experiment = exp
    num_runs = 5
    if exp == isic:
        num_runs = 3
    plot_stdev = False
    plot_eps = False
    num_clients = exp['num_clients']
    dim = exp['dimension']
    ####################################

    task = experiment['name']

    metric_names_array = ['metric', 'loss', 'error']

    for name in metric_names_array:
        metric_name = experiment[name]

        path = f'log(noise_scale)/{task}/hp_sweep_results'
        # hyperparameter_name = 'granularity'
        # hyperparameter_values = ['0.1', '0.01', '0.001', '0.0001', '0.00001', '0.000001']
        hyperparameter_name = 'noise_scale'
        # '0.00001'
        hyperparameter_values = [ '0.0001', 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        rounds_per_run = 20

        server_subdirectory = 'metrics/server_metrics.json'


        plot_dir = os.path.join('plots', task)
        os.makedirs(plot_dir, exist_ok=True)


        title = f'Hyperparameter: {hyperparameter_name} ({task})'
        image_file_name = f'{task}_{metric_name}'
        plt.xlabel('Rounds')
        plt.ylabel(metric_name)
        plt.title(title)

        for value in hyperparameter_values:
            # for plots, these are averages over runs
            y_data = []
            y_stdev = []

            print(f'--hparam: {value}--')
            experiment_folder = os.path.join(path, f'{hyperparameter_name}_{value}')
            
            # dict with keys corresponding rounds and values being an array of values across the rounds
            collective_data = {f'round_{i}': [] for i in range(1, 1+rounds_per_run)}

            # settings are the same across runs, they differ across different hyperparameter_values
            privacy_settings = {}
            for run in range(1, 1+num_runs):
                tag = f'Run{run}'
                run_folder = os.path.join(experiment_folder, tag, server_subdirectory)

                with open(run_folder) as file:
                    metrics_dict = json.load(file)
                    privacy_settings = metrics_dict['privacy_hyperparameters']
                    metric_array = metrics_dict[metric_name]
                    if metric_name == 'round vs l_inf_error':
                        if isinstance(metric_array, dict):
                            # convert dict to array
                            metric_array = list(metric_array.values())
                    i = 1
                    for metric in metric_array:
                        collective_data[f'round_{i}'].append(metric)
                        i += 1


            for val_array in collective_data.values():    
                mean = statistics.mean(val_array)
                standard_deviation = statistics.stdev(val_array)
                y_data.append(mean)
                y_stdev.append(standard_deviation)
                print(mean, standard_deviation)
            
            accountant = DistributedDiscreteGaussianAccountant(
                l2_clip = privacy_settings['clipping_threshold'],
                noise_scale = privacy_settings['noise_scale'], 
                granularity =  privacy_settings['granularity'], 
                model_dimension = dim, 
                randomized_rounding_bias = privacy_settings['bias'], 
                number_of_trustworth_fl_clients = num_clients,
                fl_rounds=rounds_per_run
            )
            eps = format(accountant.optimal_adp_epslion(), '.2e')

            label = f'{value}'
            if plot_eps:
                label=f'{value} (eps {eps})'
                
            plt.plot(y_data, marker='o', linestyle='-', label=label)
            
            if plot_stdev:
                plt.errorbar(x = list(range(1, 1+rounds_per_run)), y=y_data, yerr=y_stdev, label=f'Error {value}')


        fig_path = os.path.join(plot_dir, f'{image_file_name}.png')
        plt.legend()
        plt.savefig(fig_path)
        plt.clf() 

