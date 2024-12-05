#!/bin/bash

export DHOME="$HOME/projects/rl-radiotherapy"
export OUT="$PWD/runs"

if [ ! -e $DHOME/sjob.template ]; then
  echo "sjob.template not found"
  exit 1
fi

# NAME GPUTYPE [W]

export NAME=$1
export GPUTYPE=${2:-L40S}

if [ "$GPUTYPE" != "A40" -a "$GPUTYPE" != "L40S" ]; then
  echo "$GPUTYPE not available"
  exit 1
fi

if [ ! -z $2 ]; then 
  printf "\nJob  name: $NAME  using $GPUTYPE \n"
  mkdir -p $OUT/$CONF
  envsubst '$NAME $GPUTYPE $OUT' < $PWD/sjob.template > $OUT/$NAME/$NAME.run
  sbatch $OUT/$NAME/$NAME.run

else
  printf "\nneed NAME and GPUTYPE(A40|L40S)\n\n"

fi