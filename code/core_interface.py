"""
"""
import numpy as np
import pandas as pd
import os, sys
import core

data=None # make the Data object global here

class CoreInterface(object):
    """Simple interface to the Kerr code in godot.core
    """
    
    def __init__(self, binned_weights, min_exp=0.3):
        """
        parameters:
            binned_weights : a main.BinnedWeights object
            min_exp : float
                minimum fractional exposure accepted
        """
        global data
        data=binned_weights.data
   
        # Create list of raw cells
        # Required parameters for the core.Cell constructor
        cell_pars=dict(tstart=0,tstop=0,exposure=1,photon_times=[],photon_weights=[],
                    source_to_background_ratio=0.5)
        cells=[]; rejected=0
        
        for c in binned_weights: # loop over interval information dicts
            t, tw, fexp, w, S, B = [c[x] for x in 't tw fexp w S B'.split()]
            if fexp<min_exp:
                rejected+=1
                continue
            # note: convert times to MET using expression in core
            cell_pars.update(tstart=core.mjd2met(t-0.5*tw),
                             tstop=core.mjd2met(t+0.5*tw), 
                             photon_weights=w, 
                             exposure=S, 
                             source_to_background_ratio=S/B)
            cells.append(core.Cell(**cell_pars))

        if data.verbose>0:
            print(f'Loaded {len(cells)} cells for further analysis')
            if rejected>0:
                print(f'\trejected {rejected} with fractional exposure < {min_exp:.2f}')
        if len(cells)==0:
            raise Exception(f'Failed to find any cell with frational exposure > {min_exp}')
        
        # generate the likelihood functions for all cells
        self.cll = core.CellsLogLikelihood(cells) 

    def __repr__(self):
        return f'{self.__class__}\t {len(self.clls)} cells'
    
    def to_dataframe(self, rvals):
        """convert the rvals array, with shape (N,5) to a dataframe
        columns are:
            t: value in MJD
            tw: bin width
            flux : relative rate or 95% CL limit if error[1]==-1
            errors: 2-tuple flux error or (0,-1) 
        """
        rvt = rvals.T
        return pd.DataFrame.from_dict(
                    dict(
                         t=rvt[0], 
                         tw=2*rvt[1], 
                         flux=rvt[2].round(4),
                         errors=[ tuple(u) for u in rvals[:,3:].round(4)],
                        )  
                      )

    
    def bb_lightcurve(self, dataframe=True, **pars):
        """ Return a Bayesian blocks flux density light curve
        pars:
            tsmin=4, plot_years=False, plot_phase=False, bb_prior=8
        """
        rvals =  self.cll.get_bb_lightcurve(**pars)
        if not dataframe: return rvals
        return self.to_dataframe(rvals)
    
    def lightcurve(self, dataframe=True, **pars):
        """Return a flux density light curve for the raw cells.
        
        pars:
            tsmin=4, plot_years=False, plot_phase=False,  get_ts=False
        """
        rvals = self.cll.get_lightcurve(**pars)
        if not dataframe: return rvals
        return self.to_dataframe(rvals)
    
    def plot_bb(self, **pars):
        """tsmin=4,fignum=2,clear=True,color='C3',
            plot_raw_cells=True,bb_prior=4,plot_years=False,
            no_bb=False,log_scale=False,
            plot_phase=False,ax=None
        """
        return self.cll.plot_cells_bb(**pars)
    
    def plot_lc(self, **pars):
        return self.cll.plot_clls_lc(**pars)
