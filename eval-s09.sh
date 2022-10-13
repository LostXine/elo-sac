#!/bin/bash

if (( $# < 1 )); then
    >&2 echo "Please specify GPU ID"
    exit 1
fi

F_END=${0: -4:1}
F_START=${0: -5:1}

echo "Use GPU: $1"
for i in $(seq $F_START $F_END)
do
    echo "Seed $i"
    ./eval-backbone512.sh $1 $i
done

# python3 ../mail.py
