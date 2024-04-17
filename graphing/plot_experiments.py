import matplotlib.pyplot as plt
from data import *

plt.title('Fed-ISIC ')
plt.xlabel('FL Rounds')
plt.ylabel('balanced accuracy')


plt.plot(isic_central_001212, label='Central (stddev=0.001212)')
plt.plot(isic_central_001559, label='Central (stddev=0.001559)')
plt.plot(isic_central_001732, label='Central (stddev=0.001732)')

plt.plot(isic_distributed_0001, label='Distributed (stddev=0.0001)')
plt.plot(isic_distributed_0003, label='Distributed (stddev=0.0003)')
plt.plot(isic_distributed_0005, label='Distributed (stddev=0.0005)')





plt.legend()

plt.savefig('graphing/4_16/ISIC.pdf')