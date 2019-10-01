#!/bin/bash
# Setup for ipython or jupyterlab

# path to  anaconda
# enable if not done in .bashrc
#myconda=/nfs/farm/g/glast/u/burnett/anaconda3
#export PATH=${myconda}/bin:${PATH}

export FERMI=/nfs/farm/g/glast/g/catalog/pointlike/fermi

# set python's path to use code here (with physical links)
here=`pwd -P`
export PYTHONPATH=$here/code:$PYTHONPATH

# alias to start a jupyter lab server - note the port
# to connect from a remote computer, run ssh -N -f -L localhost:<localport>:localhost:8890 <userid>@<rhelhost>.slac.stanford.edu
# where <rhelhost> is the particular host, and <localport>, e.g., 8888 is to port for your local browser. 
alias jupyterlab='jupyter lab --notebook-dir=$here --port=8890 --no-browser'


