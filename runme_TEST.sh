#!/bin/bash

# This script is used to run the server and clients in parallel.
NUMBER_OF_CLIENTS=$2
SERVER_STRATEGY=$1

PROJECT_PATH=$(pwd)
VENV_ACTIVATE_PATH=".venv/bin/activate"
OS_TYPE=$(uname)
echo $OS_TYPE

if [[ $OS_TYPE == "Darwin" ]]; then
    osascript -e "tell application \"Terminal\" to do script \"cd $PROJECT_PATH && source $VENV_ACTIVATE_PATH && python3 src/server.py '$SERVER_STRATEGY' '$NUMBER_OF_CLIENTS'\""
    osascript -e 'tell application "Terminal" to activate'

    osascript -e "tell application \"Terminal\" to do script \"cd $PROJECT_PATH && source $VENV_ACTIVATE_PATH && python3 src/client.py airplane automobile bird cat deer\""
    osascript -e "tell application \"Terminal\" to do script \"cd $PROJECT_PATH && source $VENV_ACTIVATE_PATH && python3 src/client.py dog frog horse ship truck\""
    osascript -e 'tell application "Terminal" to activate'
    
    echo "All scripts have been started."
fi


if [[ $OS_TYPE == "Linux" ]]; then
    konsole cd $PROJECT_PATH && source $VENV_ACTIVATE_PATH && python3 src/server.py $SERVER_STRATEGY $NUMBER_OF_CLIENTS
    gnome-terminal cd $PROJECT_PATH && source $VENV_ACTIVATE_PATH && python3 src/client.py airplane automobile bird cat deer
    gnome-terminal cd $PROJECT_PATH && source $VENV_ACTIVATE_PATH && python3 src/client.py dog frog horse ship truck
    
    echo "All scripts have been started."
fi

