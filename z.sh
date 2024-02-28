#!/bin/bash

EXPERIMENT=$1


case ${EXPERIMENT} in
    "heart")
        bash research/flamby_distributed_dp/fed_heart_disease/run.sh
        ;;
    "isic")
        bash research/flamby_distributed_dp/fed_isic2019/run.sh
        ;;
    "ixi")
        bash research/flamby_distributed_dp/fed_ixi/run.sh
        ;;
    *)
        echo "$1 does not match any of heart, isic, ixi"
        ;;
esac



