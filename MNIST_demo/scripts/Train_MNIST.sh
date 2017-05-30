#!/usr/bin/env bash
# utilities
function init { 
    if [ -z "${!1}" ]; then export $1=$2; fi
}
function define { 
    export $1=$2
}
function waiting {
    for pid in "$@"
    do
        while [ -e /proc/$pid ]
        do
            sleep 1
        done
    done
}

init enableSave true
init notify "false"
init maxEpoch 50
init learningRateDecayRatio 0.5
init removeOldCheckpoints false
init optimMethod "adadelta"
init batchSize 128

function GCN4 {    
    define dataset "MNIST"
    define model "GCN"
    define CUDA_VISIBLE_DEVICES 0
    define gpuDevice "{1}"
    define savePath "logs/GCN4"
    define note "GCN4"
    define customParams "{rho=0.9,eps=1e-6,orientation=4}"
    th train.lua
}

PID=""
(GCN4;) &
PID="$PID $!"
waiting $PIDGN2
