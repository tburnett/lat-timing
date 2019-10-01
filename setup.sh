#!/bin/bash
# Setup for ipython or jupyterlab

# path to  anaconda
# enable if not done in .bashrc
#myconda=/nfs/farm/g/glast/u/burnett/anaconda3
#export PATH=${myconda}/bin:${PATH}

export FERMI=/nfs/farm/g/glast/g/catalog/pointlike/fermi
# set python to code here (with physical links)
here=`pwd -P`
export PYTHONPATH=$here/code:$PYTHONPATH

# to start a jupyter lab server - note the port
# to connect from a remote computer, run ssh -N -f -L localhost:<localport>:localhost:8890 <userid>@rhel6-64x.slac.stanford.edu
# and, on the remote computer, ssh -N -f -L localhost:8888:localhost:8890 burnett@rhel6-64X.slac.stanford.edu, where "X" is the
# particular host. 
alias jupyterlab='jupyter lab --notebook-dir=$here --port=8890 --no-browser'


