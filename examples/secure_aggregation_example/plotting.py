import matplotlib.pyplot as plt
from enum import Enum
import numpy as np
import json

FILEPATH = '.json'

with open(FILEPATH, 'r') as f:
  data = json.load(f)

class ExperimentType(Enum):
    UTILITY_EPOCH = 'utility across epoch'

class Experiments:
    def __init__(self, ml_task: str, experiment_type: ExperimentType) -> None:
        self.ml_task = ml_task
        self.experiment_type = experiment_type

        self.repetitions = 0

        self.x_lists = {}     
        self.y_lists = {}

    def append_experiment(self, x_list, y_list):
        self.repetitions += 1
        self.x_lists[self.repetitions] = x_list
        self.y_lists[self.repetitions] = y_list

    def generate_experiments(self):

        if self.repetitions == 0:
            print('no experiment has been added, aborting.')
            exit()
        id = 1

        while id <= self.repetitions:
            yield id, self.x_lists[id], self.y_lists[id]
            id += 1

def utility_epoch(experiments: Experiments) -> None:


    assert experiments.experiment_type == ExperimentType.UTILITY_EPOCH

    for id, x_list, y_list in experiments.generate_experiments():
        plt.plot(x_list, y_list, label=f'Run {id}')
    plt.legend()
    plt.show()


e = Experiments(ml_task='test', experiment_type=ExperimentType.UTILITY_EPOCH)
for i in np.arange(0.1, 3, 0.1):
    x = list(np.arange(-10, 10, 0.1))
    y = [num ** i for num in np.arange(-10, 10, 0.1)]
    e.append_experiment(x, y)

utility_epoch(e)