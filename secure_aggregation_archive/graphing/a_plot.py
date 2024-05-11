import os, json
import matplotlib.pyplot as plt

from fl4health.privacy.distributed_discrete_gaussian_accountant import DistributedDiscreteGaussianAccountant

results_dir = 'log'

def new_graph_set(name='graph', silent=True):
    dir = f'graphing/{name}'
    if not os.path.exists(dir):
        os.makedirs(dir)
        if not silent:
            print('file created: ', dir)
    else:
        if not silent:
            print('file exists: ', dir)


def read_server_metrics(training_task, dp_type, hp_type, hp_value, run_number):
    id_1 = f'{training_task}_{dp_type}'
    id_2 = f'{hp_type}_{hp_value}'
    id_3 = f'Run{run_number}'
    path = f'logs/{results_dir}/{id_1}/hp_sweep_results/{id_2}/{id_3}/metrics/server_metrics.json'
    with open(path) as file:
        return json.load(file)
    

def read_client_metrics(training_task, dp_type, hp_type, hp_value, run_number, client_number):
    id_1 = f'{training_task}_{dp_type}'
    id_2 = f'{hp_type}_{hp_value}'
    id_3 = f'Run{run_number}'
    id_4 = f'client_{client_number}_metrics'
    path = f'logs/{results_dir}/{id_1}/hp_sweep_results/{id_2}/{id_3}/metrics/{id_4}.json'
    with open(path) as file:
        return json.load(file)


def plot_run(training_task, metric_name, hp_value, dp_type, hp_type, plot_more=True):

    info = {
        'training_task': training_task, 
        'dp_type': dp_type, 
        'hp_type': hp_type, 
        'hp_value': hp_value, 
        'run_number': '1', 
    }
    subfolder_name = f"{training_task}/{metric_name}"
    
    # get data 
    try:
        metrics = read_server_metrics(**info)[metric_name]
    except KeyError:
        print('KeyError encountered while processing: ', info)
        exit()

    if dp_type == 'distributed':
        
        # single round accounting
        N = 1
        data_size = 23247
        sampling_ratio_per_round = 1
        privacy_hyperparameters = read_server_metrics(**info)['privacy_hyperparameters']

        accountant = DistributedDiscreteGaussianAccountant(
            l2_clip=privacy_hyperparameters["clipping_threshold"],
            noise_scale=privacy_hyperparameters["noise_scale"],
            granularity=privacy_hyperparameters["granularity"],
            model_dimension=privacy_hyperparameters["model_dimension"],
            randomized_rounding_bias=privacy_hyperparameters["bias"],
            number_of_trustworthy_fl_clients=privacy_hyperparameters["num_clients"],
            fl_rounds=N,
            privacy_amplification_sampling_ratio=sampling_ratio_per_round,
            approximate_dp_delta=1/data_size**2
        )

        eps_approx = accountant.fl_approximate_dp_accountant(amplify=False)
        eps_approx = round(eps_approx, 1)

        label = f"{info['hp_value']}(eps = {eps_approx})"

    # plot
    if 'label' not in locals():
        label = info['hp_value']
    plt.plot(metrics, marker='o', linestyle='-', label=label)
    
    if not plot_more:

        # label plot
        title = f"{info['training_task']}_{info['dp_type']} (hp: {info['hp_type']})"
        plt.title(title)
        plt.xlabel('FL Rounds')
        plt.ylabel(metric_name)
        plt.legend()

        # save
        new_graph_set(name=subfolder_name)
        image_name = f"{info['dp_type']}.jpeg"
        path = f"graphing/{subfolder_name}/{image_name}"
        plt.savefig(path)
        print('Figure saved to: ', path)
        plt.clf()

def heart_disease_plot_by_hp(metric_name='train_meter_FedHeartDisease_accuracy'):

    # ---------------------- Edit (start) ---------------------------------
    s_central = ' 0.01 0.008 0.009 0.011 0.012 '
    experiment_1 = {
        'dp_type': 'central',
        'hp_type': 'stdev',
        'hp_values': s_central.split()
    }

    s_distributed = ' 0.002 0.00225 0.0025 0.00275 0.0025'
    experiment_2 = {
        'dp_type': 'distributed',
        'hp_type': 'noise',
        'hp_values': s_distributed.split()
    }

    s_local = ' 0.008 0.009 0.01 0.011 '
    experiment_3 = {
        'dp_type': 'local',
        'hp_type': 'noise',
        'hp_values': s_local.split()
    }

    experiments = [experiment_1, experiment_2, experiment_3]

    training_task='fed_heart_disease'
    # ---------------------- Edit (end) ---------------------------------

    for exp in experiments:
        dp_type = exp['dp_type']
        hp_type = exp['hp_type']  
        hp_values = exp['hp_values']      

        hp_value = hp_values.pop()
        while len(hp_values) > 1:
            plot_run(training_task=training_task, metric_name=metric_name, hp_value=hp_value, dp_type=dp_type, hp_type=hp_type, plot_more=True)
            hp_value = hp_values.pop()
        plot_run(training_task=training_task,  metric_name=metric_name, hp_value=hp_value, dp_type=dp_type, hp_type=hp_type, plot_more=False)

def isic2019_plot_by_hp(metric_name='train_meter_FedIsic2019_balanced_accuracy'):

    # ---------------------- Edit (start) ---------------------------------

    s_central = ' 0.00001 '
    experiment_1 = {
        'dp_type': 'central',
        'hp_type': 'stdev',
        'hp_values': s_central.split()
    }

    s_distributed = ' '
    experiment_2 = {
        'dp_type': 'distributed',
        'hp_type': 'noise',
        'hp_values': s_distributed.split()
    }

    s_local = ' 0.1 '
    experiment_3 = {
        'dp_type': 'local',
        'hp_type': 'noise',
        'hp_values': s_local.split()
    }

    experiments = [experiment_1, experiment_2, experiment_3]
    experiments = [experiment_1, experiment_3]

    training_task = 'fed_isic2019'
    # ---------------------- Edit (end) ---------------------------------


    for exp in experiments:
        dp_type = exp['dp_type']
        hp_type = exp['hp_type']  
        hp_values = exp['hp_values']      

        hp_value = hp_values.pop()
        while len(hp_values) > 0:
            plot_run(training_task=training_task, metric_name=metric_name, hp_value=hp_value, dp_type=dp_type, hp_type=hp_type, plot_more=True)
            hp_value = hp_values.pop()
        plot_run(training_task=training_task,  metric_name=metric_name, hp_value=hp_value, dp_type=dp_type, hp_type=hp_type, plot_more=False)

def ixi_plot_by_hp(metric_name='train_meter_FedIXI_dice'):

    # ---------------------- Edit (start) ---------------------------------
    s_central = ' 0.008 0.009 0.01 0.011 0.012 '
    experiment_1 = {
        'dp_type': 'central',
        'hp_type': 'stdev',
        'hp_values': s_central.split()
    }

    s_distributed = ' 0.0027 0.003 0.0033 0.0037 0.004 '
    experiment_2 = {
        'dp_type': 'distributed',
        'hp_type': 'noise',
        'hp_values': s_distributed.split()
    }

    s_local = ' 0.008 0.009 0.01 0.011 0.012 '
    experiment_3 = {
        'dp_type': 'local',
        'hp_type': 'noise',
        'hp_values': s_local.split()
    }

    experiments = [experiment_1, experiment_2, experiment_3]

    training_task = 'fed_ixi'
    # ---------------------- Edit (end) ---------------------------------

    for exp in experiments:
        dp_type = exp['dp_type']
        hp_type = exp['hp_type']  
        hp_values = exp['hp_values']      

        hp_value = hp_values.pop()
        while len(hp_values) > 1:
            plot_run(training_task=training_task, metric_name=metric_name, hp_value=hp_value, dp_type=dp_type, hp_type=hp_type, plot_more=True)
            hp_value = hp_values.pop()
        plot_run(training_task=training_task,  metric_name=metric_name, hp_value=hp_value, dp_type=dp_type, hp_type=hp_type, plot_more=False)


def ixi_plot_by_dp_scheme(metric_name='train_meter_FedIXI_dice'):

    # ---------------------- Edit (start) ---------------------------------
    experiment_1 = {
        'dp_type': 'central_revised',
        'hp_type': 'epsilon',
        'hp_values': ['0.001']
    }

    experiment_2 = {
        'dp_type': 'central',
        'hp_type': 'epsilon',
        'hp_values': ['0.001']
    }

    experiment_3 = {
        'dp_type': 'distributed',
        'hp_type': 'noise',
        'hp_values': ['0.001']
    }

    experiment_4 = {
        'dp_type': 'local',
        'hp_type': 'noise',
        'hp_values': ['0.001']
    }
    experiments = [experiment_1, experiment_2, experiment_3, experiment_4]
    training_task = 'fed_ixi'
    run_number = 1
    # ---------------------- Edit (end) ---------------------------------
    for exp in experiments:
        for hp_value in exp['hp_values']:
            info = {
                'training_task': training_task, 
                'dp_type': exp['dp_type'], 
                'hp_type': exp['hp_type'], 
                'hp_value': hp_value, 
                'run_number': run_number}
            try:
                metrics = read_server_metrics(**info)[metric_name]
            except KeyError:
                print('KeyError encountered while processing: ', info)
                exit()

            plt.plot(metrics, marker='o', linestyle='-', label=info['dp_type'])

    # label plot
    plt.title(training_task)
    plt.xlabel('FL Rounds')
    plt.ylabel(metric_name)
    plt.legend()

    # save
    destination = f'{training_task}/{metric_name}'
    new_graph_set(name=destination)
    path = f"graphing/{destination}/_compare.jpeg"
    plt.savefig(path)
    print('Figure saved to: ', path)
    plt.clf()

def isic2019_plot_by_dp_scheme(metric_name='train_meter_FedIsic2019_balanced_accuracy'):

    # ---------------------- Edit (start) ---------------------------------
    experiment_1 = {
        'dp_type': 'central_revised',
        'hp_type': 'epsilon',
        'hp_values': ['10000']
    }

    experiment_2 = {
        'dp_type': 'central',
        'hp_type': 'epsilon',
        'hp_values': ['10']
    }

    experiment_3 = {
        'dp_type': 'distributed',
        'hp_type': 'noise',
        'hp_values': ['1']
    }

    experiment_4 = {
        'dp_type': 'local',
        'hp_type': 'noise',
        'hp_values': ['10']
    }

    experiments = [experiment_1, experiment_2, experiment_3, experiment_4]
    training_task = 'fed_isic2019'
    run_number = 1
    # ---------------------- Edit (end) ---------------------------------
    for exp in experiments:
        for hp_value in exp['hp_values']:
            info = {
                'training_task': training_task, 
                'dp_type': exp['dp_type'], 
                'hp_type': exp['hp_type'], 
                'hp_value': hp_value, 
                'run_number': run_number}
            try:
                metrics = read_server_metrics(**info)[metric_name]
            except KeyError:
                print('KeyError encountered while processing: ', info)
                exit()

            plt.plot(metrics, marker='o', linestyle='-', label=info['dp_type'])

    # label plot
    plt.title(training_task)
    plt.xlabel('FL Rounds')
    plt.ylabel(metric_name)
    plt.legend()

    # save
    destination = f'{training_task}/{metric_name}'
    new_graph_set(name=destination)
    path = f"graphing/{destination}/_compare.jpeg"
    plt.savefig(path)
    print('Figure saved to: ', path)
    plt.clf()

def heart_disease_plot_by_dp_scheme(metric_name='train_meter_FedHeartDisease_accuracy'):

    # ---------------------- Edit (start) ---------------------------------
    experiment_1 = {
        'dp_type': 'central_revised',
        'hp_type': 'epsilon',
        'hp_values': ['0.001']
    }

    experiment_2 = {
        'dp_type': 'central',
        'hp_type': 'epsilon',
        'hp_values': ['0.001']
    }

    experiment_3 = {
        'dp_type': 'distributed',
        'hp_type': 'noise',
        'hp_values': ['0.001']
    }

    experiment_4 = {
        'dp_type': 'local',
        'hp_type': 'noise',
        'hp_values': ['0.001']
    }
    experiments = [experiment_1, experiment_2, experiment_3, experiment_4]
    training_task = 'fed_heart_disease'
    run_number = 1
    # ---------------------- Edit (end) ---------------------------------
    for exp in experiments:
        for hp_value in exp['hp_values']:
            info = {
                'training_task': training_task, 
                'dp_type': exp['dp_type'], 
                'hp_type': exp['hp_type'], 
                'hp_value': hp_value, 
                'run_number': run_number}
            try:
                metrics = read_server_metrics(**info)[metric_name]
            except KeyError:
                print('KeyError encountered while processing: ', info)
                exit()

            plt.plot(metrics, marker='o', linestyle='-', label=info['dp_type'])

    # label plot
    plt.title(training_task)
    plt.xlabel('FL Rounds')
    plt.ylabel(metric_name)
    plt.legend()

    # save
    destination = f'{training_task}/{metric_name}'
    new_graph_set(name=destination)
    path = f"graphing/{destination}/_compare.jpeg"
    plt.savefig(path)
    print('Figure saved to: ', path)
    plt.clf()

if __name__ == '__main__':
    # accuracy
    heart_disease_plot_by_hp()
    isic2019_plot_by_hp()
    ixi_plot_by_hp()
    # loss
    heart_disease_plot_by_hp('loss')
    isic2019_plot_by_hp('loss')
    ixi_plot_by_hp('loss')
    # # compare accuracy across dp schemes
    # ixi_plot_by_dp_scheme()
    # isic2019_plot_by_dp_scheme()
    # heart_disease_plot_by_dp_scheme()
    # # compare loss across dp schemes
    # ixi_plot_by_dp_scheme('loss')
    # isic2019_plot_by_dp_scheme('loss')
    # heart_disease_plot_by_dp_scheme('loss')

