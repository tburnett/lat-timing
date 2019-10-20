"""
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
import core

data=None # make the Data object global here

class CoreInterface(object):
    """Simple interface to the Kerr code in godot.core
    """
    
    def __init__(self, binned_weights):
        """
        parameters:
            binned_weights : a main.BinnedWeights object
        """
        global data
        data=binned_weights.data
        interval=data.interval
   
        cell_pars=dict(tstart=0,tstop=0,exposure=1,photon_times=[],photon_weights=[],
                    source_to_background_ratio=0.5)

        cells=[]; reject=0
        for c in binned_weights:

            t, exp, w, S, B = [c[x] for x in 't exp w S B'.split()]
            if exp==0:
                reject+=1
                continue
            cell_pars.update(tstart=t-0.5*interval, tstop=t-0.5*interval, 
                             photon_weights=w, 
                             exposure=S, 
                             source_to_background_ratio=S/B)
            assert len(w)>0
            cells.append(core.Cell(**cell_pars))
        self.cll = core.CellsLogLikelihood(cells)
        if data.verbose>0:
            print(f'Loaded {len(cells)} cells for further analysis')
            if reject>0:
                print(f'\trejected {reject}')
    
    def __repr__(self):
        return f'{self.__class__}\t {len(self.cells)} cells'
    
    def lightcurve(self, dataframe=True, *pars):
        rvals =  self.cll.get_lightcurve(*pars)
        if not dataframe: return rvals
        return pd.DataFrame([rvals[:,0],rvals[:,2], rvals[:,3], ], index='time rate error'.split()).T
    
    def bb_lightcurve(self, *pars):
        return self.cll.get_bb_lightcurve(*pars)
    
    def plot_bb(self, *pars):
        return self.cll.plot_cells_bb(*pars)
    
    def plot_lc(self, *pars):
        return self.cll.plot_clls_lc(*pars)

    @property
    def cells(self):
        return self.cll.cells