#!/bin/bash

# pass in command line arg "clean" to kill all previous processes

# rm examples/secagg_non_private_example/*.pkl

# source .venv/bin/activate

mkdir -p examples/secagg_non_private_example/log

num_clients=3

clean()
{
    echo "killing processes"
    cat < examples/secagg_non_private_example/log/running_pid.txt

    PIDFile="examples/secagg_non_private_example/log/running_pid.txt"
    kill $(<"$PIDFile")
}

if [ "$1" = "clean" ]
then
    clean
elif [ "$1" = "ls" ]
then
    pidof python
elif [ "$1" = "ls-kill" ]   # be careful, this kills EVERY python process
then
    echo "finding python pid"
    pidof python
    echo "killing ALL python pid"
    for pid in $(pidof python)
    do
        kill -9 "$pid"
    done
    echo "done"
else
    echo "starting flower..."

    array=()    # PID

    nohup python -m examples.secagg_non_private_example.server > examples/secagg_non_private_example/log/server.out 2>&1 & array[${#array[@]}]=$!
    sleep 10

    for (( i=1; i<=${num_clients}; i++ ))
    do
        log_path="examples/secagg_non_private_example/log/client_${i}.out"
        nohup python -m examples.secagg_non_private_example.client --client_number="$i" > ${log_path} 2>&1 & array[${#array[@]}]=$!
    done
    echo "saving pid to file"
    echo "${array[*]}"
    echo "${array[*]}" > examples/secagg_non_private_example/log/running_pid.txt
    read -p "Press y to terminate session >>> " input
    if [ "$input" = "y" ]
    then
        clean
    fi
fi
