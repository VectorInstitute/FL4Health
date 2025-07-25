# NnUNetClient With Personalization Example

Building on the [nnunet_example](../nnunet_example/README.md), here we demonstrate how personalized
methods can be applied to `NnUNetClient` class. The config requirements remain the same as in the original
`nnunet_example`.

To run a federated learning experiment with nnunet models, first ensure you are in the FL4Health directory and then start the nnunet server using the following command. To view a list of optional flags use the --help flag

```bash
python -m examples.nnunet_pfl_example.server --config_path examples/nnunet_pfl_example/config.yaml
```

Once the server has started, start the necessary number of clients specified by the `n_clients` key in the config file. Each client can be started by running the following command in a separate session. To view a list of optional flags use the --help flag.

```bash
# ditto
python -m examples.nnunet_pfl_example.client --dataset_path examples/datasets/nnunet --personalized_strategy ditto

# mr-mtl
python -m examples.nnunet_pfl_example.client --dataset_path examples/datasets/nnunet --personalized_strategy mr_mtl
```

The same MSD dataset that was used in the original `nnunet_example` is also used here.
