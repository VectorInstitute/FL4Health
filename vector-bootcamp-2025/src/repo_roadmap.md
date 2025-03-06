# Repository Roadmap

In this document, we'll provide a brief overview of the library structure and broadly categorize the examples code by
their fit with the four lectures given in the Lab/Learn phase of the bootcamp.

### Repository Structure

#### docs/

This folder simply houses our automatically built Sphinx documentation. To access a nicely rendered version of these
docs, please visit: https://vectorinstitute.github.io/FL4Health/.

The documentation remains a work-in-progress. However, if you're interested in reading rendered documentation for the
various functions in the core library, they can be found at:
https://vectorinstitute.github.io/FL4Health/reference/api/fl4health.html.

#### examples/

This is where you'll likely spend at least some time. The `examples/` folder houses a number of demonstrations of
implementing various kinds of federated learning (FL) workflows. There are a lot of examples here.

In Section [Example Categorization](#example-categorization), we roughly organize these examples to correspond to the
various materials covered in the lectures. There are also some brief descriptions of the different examples in the
Examples [README.MD](../examples/README.MD).

Another important folder to note is `examples/utils/` which houses a small script called `run_fl_local.sh`. This is a
nice helper script that automates the process of starting up servers and clients for the examples. At present, it is
set up to run the `examples/basic_example/` code with 2 clients. It can, however, be modified to run many of the
examples and dump logs to the specified locations. To run this script, from the top level of the library one executes
```bash
bash examples/utils/run_fl_local.sh
```

#### fl4health/

The core components of the library are in the `fl4health/` folder. This is where you will find nearly all of the code
associated with the FL engine and implementations of various FL clients, servers, aggregation
strategies and other core components of FL. If you need to make custom additions, adding a metric, implementing your
own strategy, or including custom functionality, it might fit properly here, but likely can be folded into code that
you're writing to support your experiments instead.

If you're interested in understanding what's happening under the hood or debugging certain failures, you'll likely be
led into the various modules therein.

#### research/

Generally, this folder will not be a point of emphasis for the bootcamp. This folder houses some of the groups own
research on new and emerging ideas in FL. It is mainly meant to house experimentation and tinkering code that doesn't
necessarily fit into the core library at present.

#### tests/

This folder houses our unit, integration, and smoke tests meant to ensure code correctness associated with our
implementations. There may be some value in seeing how certain tests are run for different functions in understanding
the mechanics of various implementations. However, this isn't an area of the repository that is likely to be of
significant interest to participants.

### Example Categorization

In this section, the examples will be roughly grouped by where they most fit within the structure of the lectures
given during the Lab/Learn phase of the bootcamp. As a reminder, these categories are
* Introduction to FL
* Data Heterogeneity and Global Models
* Personal(ized) Federated Learning
* Beyond Better Optimization in Federated Learning

There will also be an Other category where the remainder of examples that exist beyond the scope of the material that
could be covered in the lectures given.

#### Introduction to FL

* `examples/basic_example`
* `examples/fedopt_example`
* `examples/ensemble_example`
* `examples/docker_basic_example`
* `examples/nnunet_example` (Integration with nnUnet, quite tricky to work with)

#### Data Heterogeneity and Global Models

* `examples/fedprox_example`
* `examples/scaffold_example`
* `examples/moon_example`
* `examples/feddg_ga_example`

#### Personal(ized) Federated Learning

* `examples/fl_plus_local_ft_example`
* `examples/fedper_example`
* `examples/fedrep_example`
* `examples/apfl_example`
* `examples/fenda_example`
* `examples/ditto_example`
* `examples/mr_mtl_example`
* `examples/fenda_ditto_example`
* `examples/perfcl_example`
* `examples/fedbn_example`
* `examples/fedpm_example`
* `examples/dynamic_layer_exchange_example`
* `examples/sparse_tensor_partial_exchange_example`

#### Beyond Better Optimization in Federated Learning

* `examples/fedpca_examples`
* `examples/feature_alignment_example`
* `examples/ae_examples/cvae_dim_example`
* `examples/ae_examples/cvae_examples`
* `examples/ae_examples/fedprox_vae_example`
* `examples/dp_fed_examples/client_level_dp`
* `examples/dp_fed_examples/client_level_dp_weighted`
* `examples/dp_fed_examples/instance_level_dp`
* `examples/dp_scaffold_example`

#### Other

* `examples/model_merge_example`
* `examples/warm_up_example/fedavg_warm_up`
* `examples/warm_up_example/warmed_up_fedprox`
* `examples/warm_up_example/warmed_up_fenda`
* `examples/fedsimclr_example`
* `examples/flash_example`
