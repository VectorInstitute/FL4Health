import matplotlib.pyplot as plt
import os, json, statistics


def assemble_path(dp_type, hp_name='noise', hp_value='0.001', task_name='fed_heart_disease', run=1, revised=True):
    prefix = f'logs/results/{task_name}_{dp_type}/hp_sweep_results/{hp_name}_{hp_value}'
    if revised:
        prefix = f'logs/results/{task_name}_{dp_type}_revised/hp_sweep_results/{hp_name}_{hp_value}'
    suffix = 'metrics/server_metrics.json'
    return f'{prefix}/Run{run}/{suffix}'

def get_metric_array(path, metric_name):
    with open(path) as file:
        data_dict = json.load(file)
        return data_dict[metric_name]

def average_metric_over_runs(dp_type, metric_name, hp_name='noise', hp_value='0.1', task_name='fed_heart_disease', num_runs=5, revised=True):

    paths = [assemble_path(dp_type, hp_name, hp_value, task_name, run=i, revised=revised) for i in range(1, 1+num_runs)] 
    raw_data = [get_metric_array(path, metric_name) for path in paths]
    min_length = min(len(table) for table in raw_data)

    averaged_data = []
    for round in range(min_length):
        mean = statistics.mean(array[round] for array in raw_data)
        averaged_data.append(mean)

    return averaged_data

def create_new_dir(name='graphs'):
    dir = f'graphing/{name}'
    if not os.path.exists(dir):
        os.makedirs(dir)



if __name__ == '__main__':
    create_new_dir(name='graphs')

    for hp_value in ['0.001', '0.01', '0.005', '0.05', '0.5', '1']:
        args = {
            "dp_type": 'distributed',
            "metric_name": 'train_meter_FedHeartDisease_accuracy',
            "hp_name": 'noise',
            'hp_value': hp_value, 
            'task_name': 'fed_heart_disease', 
            'num_runs': 5,
            'revised': False

        }
        file_name = f"{args['task_name']}_{args['dp_type']}_{args['hp_name']}_{args['hp_value']}"
        if args['revised']:
            file_name = f"{args['task_name']}_{args['dp_type']}_revised_{args['hp_name']}_{args['hp_value']}"
        path = f'{dir}/{file_name}.jpeg'
        plt.xlabel('Rounds')
        plt.ylabel('Accuracy')

        # data to plot
        y_data = average_metric_over_runs(**args)
        plt.clf()
        plt.plot(y_data, marker='o', linestyle='-', label=hp_value)
        plt.legend()
        plt.savefig(path)
        print('Figure saved to: ', path)


