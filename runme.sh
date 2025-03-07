#!/bin/bash

# This script is used to run the server and clients in parallel.
NUMBER_OF_CLIENTS=$2
SERVER_STRATEGY=$1

PROJECT_PATH=$(pwd)
VENV_ACTIVATE_PATH=".venv/bin/activate"
OS_TYPE=$(uname)
echo $OS_TYPE
CIFAR10_CLASSES=("airplane" "automobile" "bird" "cat" "deer" "dog" "frog" "horse" "ship" "truck")

if [[ $OS_TYPE == "Darwin" ]]; then
    echo "Entered loop"
    osascript -e "tell application \"Terminal\" to do script \"cd $PROJECT_PATH && source $VENV_ACTIVATE_PATH && python src/server.py '$SERVER_STRATEGY' '$NUMBER_OF_CLIENTS'\""
    osascript -e 'tell application "Terminal" to activate'

    for ((i=1 ; i<=NUMBER_OF_CLIENTS ; i++)); do
        rand_num_classes=$(( (RANDOM % 10) + 1 ))

        selected_classes=()
        while [ ${#selected_classes[@]} -lt $rand_num_classes ]; do
            rand_class=${CIFAR10_CLASSES[$((RANDOM % 10))]}

            if [[ ! " ${selected_classes[@]} " =~ " ${rand_class} " ]]; then
                selected_classes+=("$rand_class")
            fi
        done
        osascript -e "tell application \"Terminal\" to do script \"cd $PROJECT_PATH && source $VENV_ACTIVATE_PATH && python src/client.py ${selected_classes[*]}\""
        osascript -e 'tell application "Terminal" to activate'
    done
fi
echo "All scripts have been started."