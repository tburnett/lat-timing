#!/bin/bash
# Setup for ipython or jopyterlab

# path to  anaconda
myconda=/nfs/farm/g/glast/u/burnett/anaconda2

#use  anaconda for executables 
export PATH=${myconda}/bin:${PATH}

# to start a jupyter lab server - note the port
# to connect from a remote computer, run ssh -N -f -L localhost:<localport>:localhost:8890 <userid>@rhel6-64x.slac.stanford.edu
alias jupyterlab='jupyter lab --notebook-dir=/nfs/farm/g/glast/u/burnett/analysis/agn_timing --port=8890 --no-browser'

# should prevent core dumps when run in batch
ulimit -c 0

if [ "$PS1" ]; then
  echo "CONDA :" $myconda
  alias
fi
