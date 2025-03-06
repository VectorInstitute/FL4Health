# Common Issues and Troubleshooting

Because FL relies on communication between distributed processes, even if they are simulated on the same machine,
things can go a bit haywire if the communication orchestration gets off track. In this document, we'll try to list
a few of the common issues one might run into when working with the library and running experiments.

### Server and Clients Stuck and Doing Nothing

If this is happening there are several common causes for the hanging processes.

#### Not Enough Clients Have Started

A critical parameter in the configuration files is `n_clients`. See, for example,
[examples/basic_example/config.yaml](../examples/basic_example/config.yaml). In many of our examples, this parameter
is used to set the `min_fit_clients` and `min_evaluate_clients` for the strategy objects. See, for example,
[examples/basic_example/server.py](../examples/basic_example/server.py). This tells the server that it should wait for
at least `n_clients` before beginning federated learning.

If you have only started 3 clients, but `n_clients: 4`, the server (and the existing clients) will wait until at least
one more client has reported into the server before starting.

#### Ghost or Orphaned Processes Remain Running

The FL4Health library relies on the communication layer provided by Flower in order to orchestrate information
exchange between the server and client processes. While this process is generally robust, it can run aground in
certain scenarios. If FL concludes cleanly, the server and client processes will be shutdown automatically. We also
have functionality that can be used to terminate such processes when the server receives an exceptions or multiple
exceptions from participating clients if `accept_failures=False` for the server class.

However, in certain scenarios, such as (ctrl+c) stopping a process or a failure before clients have registered to the
server, processes may be left running in the background. This is especially true if you're launching processes with
`nohup`, as is done in the `examples/utils/run_fl_local.sh` script. Because these orphaned processes will still be
listening on the local IP and a specified port, they can interfere with communication of new processes that
you start with the same IP and port specifications.

To alleviate this, you need to terminate these running processes before starting any new runs. The easiest way to do
this is through `top/htop` via the terminal on Mac/Linux machines. An analogous process should be followed on windows
machines to shut such processes down.

### Scary Warnings On Startup

On starting up the server and client processes, sometimes various warnings that look a bit scary front the log files.
An example of one of these warnings appears below when running locally on CPU.

```sh
2024-11-29 08:54:04.123569: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized
to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler
flags.
/usr/local/anaconda3/envs/fl4health/lib/python3.10/site-packages/threadpoolctl.py:1214: RuntimeWarning:
Found Intel OpenMP ('libiomp') and LLVM OpenMP ('libomp') loaded at
the same time. Both libraries are known to be incompatible and this
can cause random crashes or deadlocks on Linux when loaded in the
same Python program.
Using threadpoolctl may cause crashes or deadlocks. For more
information and possible workarounds, please see
    https://github.com/joblib/threadpoolctl/blob/master/multiple_openmp.md
```

While the above warnings might appear problematic, they are often harmless and pop out from various libraries
leveraged under the hood to warn users of issues that might arise under certain conditions or provide them a chance
to install pieces of software. For example, the first output is saying that performance on the CPU could be improved
with the right tensorflow compilation. However, it isn't necessary to run the code properly.
