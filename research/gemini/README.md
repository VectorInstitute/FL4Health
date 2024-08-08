## Codebase for GEMINI experiments.

This folder contains all of the training and model scripts for the GEMINI experiments reported in the [paper](https://arxiv.org/pdf/2309.16825)
```
Tavakoli, F.; Emerson, D. B.; Ayromlou, S.; Jewell, J. T.; Krishnan, A.; Zhang, Y.; Verma, A.; and Razak, F. 2024. A Comprehensive View of Personalized Federated Learning on Heterogeneous Clinical Datasets. In Machine Learning for Healthcare 2024, 1–35
```
As the training data associated with the GEMINI experiments is private and sensitive, the data preprocessing and construction code is omitted in accordance with the policies of GEMINI.

**Note**: These scripts are not runnable outside of the GEMINI environment. They are for exposition purposes and to aid reproducibility for those with access to the GEMINI HPC system and data.

You can find the scripts and commands to run the experiments in each algorithm folder. This codebase is built using two different versions of the FL4Health library, therefore, there is a `first_requirements.txt` that should be activated for all the methods except `FedPer`, `PerFCL`, `Ditto`, and `MOON`. The modules specified in `second_requirements.txt` should be activated to run `FedPer`, `PerFCL`, `Ditto`, and `MOON`.

## Delirium extreme heterogeneity experiments
The experiments, by default, use naturally heterogeneous data splits. To perform extreme heterogeneity experiments, use `300` as the model's first layer size instead of `8093` when defining the model. This change can be easily applied by setting this parameter in model constructor. Don't forget to apply the change in both `client.py` and `server.py`. Also, the path to the data source should be adjusted to `data_path = Path("heterogeneous_data")`
