#!/usr/bin/env bash

SCRIPT=$1
if [[ -z $SCRIPT ]]
then
    echo "ERROR: Missing python script"
    exit 1
fi

if [[ ! $SCRIPT == *.py ]]
then
    echo "ERROR: Script $SCRIPT should be a python file"
    exit 1
fi

if [[ ! -f $SCRIPT ]]
then
    echo "ERROR: Script $SCRIPT does not exist"
    exit 1
fi

until python $SCRIPT; do
    echo ""
    echo "************************************************************************"
    echo "WARNING: Script crashed, restarting..." >&2
    echo "************************************************************************"
    sleep 5
    clear >$(tty)
done