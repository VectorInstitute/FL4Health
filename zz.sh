#!/bin/bash
arg1=$1
shift 1
arr=("$@")

dosomething () {

for element in "${arr[@]}"; do
    echo "$element"
done

echo "${arr[@]}"
}

dosomething 