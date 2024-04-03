#!/bin/bash

EXPERIMENT=$1

cancel_present_jobs=1


central_experiments () {

    if [[ $cancel_present_jobs == 1 ]]; then
        scancel --me
        squeue --me
    fi

    HYPERPARAMETER_NAME="epsilon"

    HYPERPARAMETER_VALUES_HEART=( 1 2 )
    bash research/flamby_central_dp/fed_heart_disease/run.sh $HYPERPARAMETER_NAME "${HYPERPARAMETER_VALUES_HEART[@]}"

    HYPERPARAMETER_VALUES_ISIC=( 1 2 )
    bash research/flamby_central_dp/fed_isic2019/run.sh $HYPERPARAMETER_NAME "${HYPERPARAMETER_VALUES_ISIC[@]}"

    HYPERPARAMETER_VALUES_IXI=( 1 2 )
    bash research/flamby_central_dp/fed_ixi/run.sh $HYPERPARAMETER_NAME "${HYPERPARAMETER_VALUES_IXI[@]}"


}

local_experiments () {

    if [[ $cancel_present_jobs == 1 ]]; then
        scancel --me
        squeue --me
    fi

    HYPERPARAMETER_NAME="noise_multiplier"

    HYPERPARAMETER_VALUES_HEART=( 1 2 )
    bash research/flamby_local_dp/fed_heart_disease/run.sh $HYPERPARAMETER_NAME "${HYPERPARAMETER_VALUES_HEART[@]}"

    HYPERPARAMETER_VALUES_ISIC=( 1 2 )
    bash research/flamby_local_dp/fed_isic2019/run.sh $HYPERPARAMETER_NAME "${HYPERPARAMETER_VALUES_ISIC[@]}"

    HYPERPARAMETER_VALUES_IXI=( 1 2 )
    bash research/flamby_local_dp/fed_ixi/run.sh $HYPERPARAMETER_NAME "${HYPERPARAMETER_VALUES_IXI[@]}"



}

distributed_experiments () {

    if [[ $cancel_present_jobs == 1 ]]; then
        scancel --me
        squeue --me
    fi

    HYPERPARAMETER_NAME="noise_scale"

    HYPERPARAMETER_VALUES_HEART=( 1 2 )
    bash research/flamby_distributed_dp/fed_heart_disease/run.sh $HYPERPARAMETER_NAME "${HYPERPARAMETER_VALUES_HEART[@]}"

    HYPERPARAMETER_VALUES_ISIC=( 1 2 )
    bash research/flamby_distributed_dp/fed_isic2019/run.sh $HYPERPARAMETER_NAME "${HYPERPARAMETER_VALUES_ISIC[@]}"

    HYPERPARAMETER_VALUES_IXI=( 1 2 )
    bash research/flamby_distributed_dp/fed_ixi/run.sh $HYPERPARAMETER_NAME "${HYPERPARAMETER_VALUES_IXI[@]}"

      
}

case ${EXPERIMENT} in
    "all")
        central_experiments
        local_experiments
        distributed_experiments
    ;;
    "central")
        central_experiments
    ;;
    "local")
        local_experiments
    ;;
    "distributed")
        distributed_experiments
    ;;
    *)
    echo "$1 is invalid argument"
    ;;
esac


