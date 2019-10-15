#!/bin/bash
# Setup and start a  jupyterlab server

# path to  anaconda
# enable if not done in .bashrc
#myconda=/nfs/farm/g/glast/u/burnett/anaconda3
#export PATH=${myconda}/bin:${PATH}

# path to pointlike data setup
export FERMI=/nfs/farm/g/glast/g/catalog/pointlike/fermi
# for P8R3_V2 irfs
export CALDB=/afs/slac/g/glast/groups/canda/irfs/p8_merit/P8R3_V2/CALDB

# set python path to include code here (with physical links)
here=`pwd -P`
export PYTHONPATH=$here/code:$PYTHONPATH

# starting a jupyter lab server - note the port
# I use remoteport=8890 instead of the default 8888 since other users may have servers running

remoteport=8890
if [ "$#" -gt  0 ]; then
    echo ${1}
    remoteport=${1}
fi
host=`hostname`
echo Starting jupyterlab with port $remoteport on $host
rm -f nohup.out
nohup jupyter lab --notebook-dir=$here --port=$remoteport --no-browser


# to connect from a remote computer, run ssh -N -f -L localhost:$localport:localhost:$remoteport $userid@rhel6-64x.slac.stanford.edu
# I set my localport to correspond to the server machine, to keep track of multiple servers

echo "To connect from a remote computer, run:"
user=`whoami`
echo "ssh -N -f -L localhost:${remoteport}:${host}.slac.stanford.edu:${remoteport} ${user}@${host}.slac.stanford.edu"
