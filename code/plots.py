""" Various useful plots
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalized_rate(data, step=None, min_exposure_factor=0.3, data_cut=None):

    # get binned data, cut out low exposure bins
    dfm = data.count_binner(step=step,cut=data_cut)
    exp =    dfm.exp.values
    exp_mean = exp.mean()
    mask = exp > min_exposure_factor*exp_mean
    interval = step or data.interval
    if data.verbose>0:
        print(f'exposure minimum factor, {min_exposure_factor}, removes {sum(~mask)}/{len(mask)} intervals')
        
    dfm = dfm.loc[mask,:]
    rel_exp = dfm.exp.values/exp_mean
    t=       dfm.time.values
    counts = dfm.counts.values
    ratio = counts/rel_exp
    y = ratio/ratio.mean()


    fig, (ax1,ax2)= plt.subplots(2,1, figsize=(15,5), sharex=True,gridspec_kw=dict(hspace=0) )
    ax1.plot(t, rel_exp, '+'); ax1.grid(alpha=0.5)
    ax1.set(ylabel=f'Exposure per {interval} day', ylim=(0,None))
    ax1.text(0.01, 0.05, f'mean exposure: {exp_mean/interval:.2e} / day',
                    transform=ax1.transAxes)
    dy = y/np.sqrt(counts) 
    ax2.errorbar(t, y, yerr=dy,  fmt='+');
    ax2.set(xlabel=r'$\mathrm{MJD}$', ylabel='Relative flux')
    ax2.axhline(1, color='grey')
    ax2.grid(alpha=0.5)
    fig.suptitle(f'Normalized flux for {data.source_name}')