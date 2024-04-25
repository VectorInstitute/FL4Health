import matplotlib.pyplot as plt
import json 

path = 'logs/log/fed_isic2019_distributed/hp_sweep_results/noise_0.00000001/Run1/metrics/server_metrics.json'
with open(path) as file:
    d = json.load(file)
# plt.plot(d['train_meter_FedIsic2019_balanced_accuracy'], label='train')
# plt.plot(d['val_meter_FedIsic2019_balanced_accuracy'], label='validation')
plt.plot(d['loss'])


plt.title('FedISIC Distributed')
plt.xlabel('FL rounds')
plt.ylabel('loss')
# plt.ylabel('balanced accuracy')


# plt.legend()

plt.savefig('graphing/Distributed_ISIC.pdf')