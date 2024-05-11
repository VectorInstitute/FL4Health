#!/bin/bash

EXPERIMENT=$1

cancel_present_jobs=0
lightweight=0
clean_up=1
erase_checkpoints=1

if [[ $clean_up == 1 ]]; then
    clear
    echo removing logs and cancelling jobs ... 
    scancel --me
    sleep 3
    rm -rf /scratch/ssd004/scratch/xuejzhao/log_error/ logs/
    squeue --me
fi

if [[ $erase_checkpoints == 1 ]]; then
    bash secure_aggregation_clean.sh
fi

central_experiments () {

    if [[ $cancel_present_jobs == 1 ]]; then
        scancel --me
        squeue --me
    fi

    HYPERPARAMETER_NAME="stdev"

    # # client is 4
    # HYPERPARAMETER_VALUES_HEART=( 0.0002 0.0006 0.001 0.0014 0.0018 )
    # bash research/flamby_central_dp/fed_heart_disease/run.sh $HYPERPARAMETER_NAME "${HYPERPARAMETER_VALUES_HEART[@]}"

    if [[ $lightweight == 0 ]]; then
        # client is 3
        HYPERPARAMETER_VALUES_ISIC=( 0.0001732 0.0005196 0.0008660 0.001212 0.001559)
        HYPERPARAMETER_VALUES_ISIC=(0.000001)
        bash research/flamby_central_dp/fed_isic2019/run.sh $HYPERPARAMETER_NAME "${HYPERPARAMETER_VALUES_ISIC[@]}"

        # client is 3
        # HYPERPARAMETER_VALUES_IXI=( 0.0001732 0.0005196 0.0008660 0.001212 0.001559)
        # bash research/flamby_central_dp/fed_ixi/run.sh $HYPERPARAMETER_NAME "${HYPERPARAMETER_VALUES_IXI[@]}"

        # # client is 4
        # HYPERPARAMETER_VALUES_TCGA_BRCA=( 0.0002 0.0006 0.001 0.0014 0.0018 )
        # bash research/flamby_central_dp/fed_tcga_brca/run.sh $HYPERPARAMETER_NAME "${HYPERPARAMETER_VALUES_TCGA_BRCA[@]}"

    fi
}

local_experiments () {

    if [[ $cancel_present_jobs == 1 ]]; then
        scancel --me
        squeue --me
    fi

    HYPERPARAMETER_NAME="noise_multiplier"

    # # client is 4
    # HYPERPARAMETER_VALUES_HEART=( 0.0001 0.0003 0.0005 0.0007 0.0009 )
    # bash research/flamby_local_dp/fed_heart_disease/run.sh $HYPERPARAMETER_NAME "${HYPERPARAMETER_VALUES_HEART[@]}"

    if [[ $lightweight == 0 ]]; then
        HYPERPARAMETER_VALUES_ISIC=( 0.0001 0.0003 0.0005 0.0007 0.0009 )
        HYPERPARAMETER_VALUES_ISIC=( 0.000001 )
        bash research/flamby_local_dp/fed_isic2019/run.sh $HYPERPARAMETER_NAME "${HYPERPARAMETER_VALUES_ISIC[@]}"

        # # client is 3
        # HYPERPARAMETER_VALUES_IXI=( 0.0001 0.0003 0.0005 0.0007 0.0009 )
        # bash research/flamby_local_dp/fed_ixi/run.sh $HYPERPARAMETER_NAME "${HYPERPARAMETER_VALUES_IXI[@]}"

        # HYPERPARAMETER_VALUES_TCGA_BRCA=( 0.0001 0.0003 0.0005 0.0007 0.0009 )
        # bash research/flamby_local_dp/fed_tcga_brca/run.sh $HYPERPARAMETER_NAME "${HYPERPARAMETER_VALUES_TCGA_BRCA[@]}"
    fi


}

distributed_experiments () {

    if [[ $cancel_present_jobs == 1 ]]; then
        scancel --me
        squeue --me
    fi

    HYPERPARAMETER_NAME="noise_scale"

    # # client is 4
    # HYPERPARAMETER_VALUES_HEART=( 0.0001 0.0003 0.0005 0.0007 0.0009 )
    # bash research/flamby_distributed_dp/fed_heart_disease/run.sh $HYPERPARAMETER_NAME "${HYPERPARAMETER_VALUES_HEART[@]}"

    if [[ $lightweight == 0 ]]; then
        # client is 3
        HYPERPARAMETER_VALUES_ISIC=( 0.0001 0.0003 0.0005 0.0007 0.0009 )
        HYPERPARAMETER_VALUES_ISIC=( 0.00000002 )
        bash research/flamby_distributed_dp/fed_isic2019/run.sh $HYPERPARAMETER_NAME "${HYPERPARAMETER_VALUES_ISIC[@]}"

        # client is 3
        # HYPERPARAMETER_VALUES_IXI=( 0.0001 0.0003 0.0005 0.0007 0.0009 )
        # bash research/flamby_distributed_dp/fed_ixi/run.sh $HYPERPARAMETER_NAME "${HYPERPARAMETER_VALUES_IXI[@]}"

        # # client is 4
        # HYPERPARAMETER_VALUES_TCGA_BRCA=( 0.0001 0.0003 0.0005 0.0007 0.0009 )
        # bash research/flamby_distributed_dp/fed_tcga_brca/run.sh $HYPERPARAMETER_NAME "${HYPERPARAMETER_VALUES_TCGA_BRCA[@]}"

    fi
      
}

just_isic () { 

    if [[ $cancel_present_jobs == 1 ]]; then
        scancel --me
        squeue --me
    fi

    # HYPERPARAMETER_NAME_CENTRAL="stdev"
    # HYPERPARAMETER_VALUES_ISIC_CENTRAL=( 0.00001 )
    # bash research/flamby_central_dp/fed_isic2019/run.sh $HYPERPARAMETER_NAME_CENTRAL "${HYPERPARAMETER_VALUES_ISIC_CENTRAL[@]}"

    # HYPERPARAMETER_NAME_DISTRIBUTED="noise_scale"
    # HYPERPARAMETER_VALUES_ISIC_DISTRIBUTED=( 0.00001 )
    # bash research/flamby_distributed_dp/fed_isic2019/run.sh $HYPERPARAMETER_NAME_DISTRIBUTED "${HYPERPARAMETER_VALUES_ISIC_DISTRIBUTED[@]}"

    # HYPERPARAMETER_NAME_LOCAL="noise_multiplier"
    # HYPERPARAMETER_VALUES_ISIC_LOCAL=( 0.000011 )
    # bash research/flamby_local_dp/fed_isic2019/run.sh $HYPERPARAMETER_NAME_LOCAL "${HYPERPARAMETER_VALUES_ISIC_LOCAL[@]}"

}

just_tcga_brca () {
    if [[ $cancel_present_jobs == 1 ]]; then
        scancel --me
        squeue --me
    fi
    HYPERPARAMETER_NAME_CENTRAL="stdev"
    HYPERPARAMETER_VALUES_CENTRAL=( 0.000001 )

    bash research/flamby_central_dp/fed_tcga_brca/run.sh $HYPERPARAMETER_NAME_CENTRAL "${HYPERPARAMETER_VALUES_CENTRAL[@]}"

    # HYPERPARAMETER_NAME_DISTRIBUTED="noise_scale"
    # HYPERPARAMETER_VALUES_DISTRIBUTED=( 0.01 )
    # bash research/flamby_distributed_dp/fed_tcga_brca/run.sh $HYPERPARAMETER_NAME_DISTRIBUTED "${HYPERPARAMETER_VALUES_DISTRIBUTED[@]}"

    HYPERPARAMETER_NAME_LOCAL="noise_multiplier"
    HYPERPARAMETER_VALUES_LOCAL=( 0.000001 )
    bash research/flamby_local_dp/fed_tcga_brca/run.sh $HYPERPARAMETER_NAME_LOCAL "${HYPERPARAMETER_VALUES_LOCAL[@]}"

}

custom () {

    # HYPERPARAMETER_NAME="stdev"
    # HYPERPARAMETER_VALUES_ISIC=( 0.0001 )
    # bash research/flamby_central_dp/fed_isic2019/run.sh $HYPERPARAMETER_NAME "${HYPERPARAMETER_VALUES_ISIC[@]}"

    # HYPERPARAMETER_NAME_DISTRIBUTED="noise_scale"
    # HYPERPARAMETER_VALUES_DISTRIBUTED=( 0.0001 )
    # bash research/flamby_distributed_dp/fed_tcga_brca/run.sh $HYPERPARAMETER_NAME_DISTRIBUTED "${HYPERPARAMETER_VALUES_DISTRIBUTED[@]}"

    # HYPERPARAMETER_NAME_LOCAL="noise_multiplier"
    # HYPERPARAMETER_VALUES_LOCAL=( 0.0001 )
    # bash research/flamby_local_dp/fed_tcga_brca/run.sh $HYPERPARAMETER_NAME_LOCAL "${HYPERPARAMETER_VALUES_LOCAL[@]}"

    # central_experiments
    # local_experiments
    distributed_experiments
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
    "just-isic")
        just_isic
    ;;
    "just-tcga-brca")
        just_tcga_brca
    ;;
    "custom")
        custom
    ;;
    "erase-checkpoints")
        bash secure_aggregation_clean.sh
    ;;
    *)
    echo "$1 is invalid argument"
    ;;
esac


if [[ $lightweight == 1 ]]; then
    echo 
    echo
    echo ">>>     lightweight mode runs fed heart, not isic and ixi"
    echo 
    echo
fi
