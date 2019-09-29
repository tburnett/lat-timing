#!/bin/bash
# Setup for ipython or jopyterlab

# if for anaconda
export MYCONDA=/nfs/farm/g/glast/u/burnett/anaconda2

# pointlike uses $FERMI to find default paths for data, diffuse, catalog files
export FERMI=$pointlike/fermi


#optionally override pointlike/python/uw
#export PYTHONPATH=.:${PYTHONPATH}

#special to use current anaconda
export PATH=${MYCONDA}/bin:${PATH}

# use a local matplotlib configuration. Important for batch to use agg at least
#export MPLCONFIGDIR=$pointlike/.matplotlib


# should prevent core dumps when run in batch
ulimit -c 0

if [ "$PS1" ]; then
  echo "MYPYTHON :" $mypython
fi
