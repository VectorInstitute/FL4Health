# Delirium prediction

Train:
```
./local_training/run_hp_sweep.sh "local_training/local_results/" "delirium" 6 200 64
```


# Mortality prediction

Train:
```
chmod +x local_training/run_hp_sweep.sh
./local_training/run_hp_sweep.sh "local_training/7_clients/" "mortality" 7 100 64
```
