#!/bin/bash

# pass in command line arg "clean" to kill all previous processes

# rm examples/secure_aggregation_example/*.pkl

source .venv/bin/activate

mkdir -p examples/secure_only_example/log

num_clients=3

clean()
{
    echo "killing processes"
    cat < examples/secure_only_example/log/running_pid.txt

    PIDFile="examples/secure_only_example/log/running_pid.txt"
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

    nohup python -m examples.secure_only_example.server > examples/secure_only_example/log/server.out 2>&1 & array[${#array[@]}]=$!
    sleep 10

    for (( i=1; i<=${num_clients}; i++ ))
    do
        log_path="examples/secure_only_example/log/client_${i}.out"
        nohup python -m examples.secure_only_example.client > ${log_path} 2>&1 & array[${#array[@]}]=$!
    done
    echo "saving pid to file"
    echo "${array[*]}"
    echo "${array[*]}" > examples/secure_only_example/log/running_pid.txt
    read -p "Press y to terminate session >>> " input
    if [ "$input" = "y" ]
    then
        clean
    fi
fi



# echo "killing processes"
# for i in "${array[@]}"
# do
#     echo "bye $i"
#     kill -9 "$i"
# done
# echo "done"
