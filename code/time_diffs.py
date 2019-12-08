"""
Code to apply the Atwood time-differencing algorithm
"""

import os, sys
import numpy as np
import pandas as pd
from scipy import random
import matplotlib.pyplot as plt
import keyword_options # a copy in the same folder to avoid getting all Fermi code


class TimeDiff(object):
    """Simulation and analysis for AGN QPO search
    """

    defaults=(
        ('viewing_period', 1000, 'number of days total'), 

        'Simulation',
        ('noise_rate'  ,0.4,   'noise photons/day'), 
        ('agn_period'  , 50.,  'QPO period in days'),  
        ('phase_width' , 0.1, 'spread in phase'),
        ('signal_count', 25,   'average number of counts per cycle'),

        'Analysis',
        ('window_size' , 100,  'window size in days'),
        ('max_freq'    , 0.05,  'maximum frequency day**-1'),

        'Processing options',
        ('make_plots',  True, 'generate plots for all steps'),
        ('binsize',     10,   'Binsize (in days) for time plots')
    )

    @keyword_options.decorate(defaults)
    def __init__(self, dataset=None, **kwargs):
        """

        """
        keyword_options.process(self, kwargs)
        
        # set up data, either real of run a simulation
        self.simulated = dataset is None
        self.dataset = self.simulate() if self.simulated else dataset 

        self.use_weights = 'weight' in self.dataset.columns
        if self.make_plots:
            plt.rc('font', size=12)
            

    def simulate(self):

        background = np.cumsum( random.exponential(scale=1/self.noise_rate, 
                                size=int(self.viewing_period*self.noise_rate*1.05)))
        signal = np.array([])
        for i in range(int(self.viewing_period/self.agn_period)):
            counts = random.poisson(lam=self.signal_count)
            delta_t = random.normal(i+0.5, self.phase_width, size=counts) * self.agn_period
            signal = np.append(signal, delta_t)
        time_data = np.sort(np.append(background,signal))
        
        if self.make_plots:

            nbins = self.viewing_period/self.binsize
            fig,ax = plt.subplots(figsize=(10,3), gridspec_kw=dict(right=0.95))
            hkw = dict(bins=np.linspace(0, self.viewing_period, nbins), histtype='step',lw=2)
            ax.hist(time_data, label='all photons', **hkw )
            ax.hist(signal,     label='signal', **hkw);
            ax.set(xlabel='Elapsed time [day]', ylabel='counts / {} days'.format(self.binsize))
            ax.legend();
            ax.set_title('Simulation: {} background and {} signal photons'.format(
                len(background),len(signal)),loc='left')

        return pd.DataFrame([time_data], index=['time']).T

    def __call__(self, window_size=None, max_freq=None, **kwargs):
        """ Run the time-differencing, then FFT
        """
        
        if window_size is None: window_size=self.window_size
        if max_freq is None: max_freq=self.max_freq
            
        fft_size = 2 * int(np.floor(window_size * max_freq))

        # generate time differences
        time_data=self.dataset.time.values
        if self.use_weights:
            weights = self.dataset.weight.values
        td = np.zeros(fft_size)
        for i1,t1 in enumerate(time_data):
            b = np.searchsorted(time_data, ( t1+window_size))
            t2 = time_data[i1+1:b]
            fb = np.floor((t2-t1)*max_freq*2).astype(int)
            td[fb] += weights[i1]*weights[i1+1:b] if self.use_weights else 1
        
        # run FFT
        norm = np.sum(np.absolute(td)/2.0, dtype=np.float32) 
        output_array = np.fft.rfft(td)
        power =np.square(np.absolute(output_array)) / norm

        if self.make_plots:
            
            if self.simulation:
                fig,(ax1,ax2) = plt.subplots(2,1, figsize=(10,8), gridspec_kw=dict(
                    hspace=0.3,top=0.92,left=0.05)  )

                time_bin = window_size/fft_size
                times=np.linspace(0.5, fft_size-0.5, fft_size)*time_bin
                ax1.plot(times, td, '+');
                ax1.grid(alpha=0.5);
                ax1.set(xlabel='time difference [days]', ylim=(0,None), 
                        ylabel='Entries per {:.1f} day'.format(time_bin))
                fig.suptitle('Processing: window size={}, max_freq={}'.format(
                    window_size, max_freq),  ha='right')
            else:
                fig, ax2 = plt.subplots(1,1, figsize=(8,3))
            
                deltaf= max_freq/fft_size*2
                freq = np.linspace(0.5, fft_size//2-0.5, fft_size//2)*deltaf
                ax2.plot( freq, power[1:], 'o', label='')
                ax2.grid(alpha=0.5);
                ax2.set(xlabel='Frequency [1/day]', ylabel='Power',yscale='log', **kwargs)
                ax2.text(0.7,0.8, 
                         f'window size {window_size}\n'\
                         f'max_freq    {max_freq}',
                         fontdict=dict(family='monospace', size=12),
                         transform=ax2.transAxes)
                if self.simulated:
                    ax2.axvline(1/self.agn_period, color='red', ls=':', label='simulated frequency')
                    ax2.legend()

                fig.set(facecolor='white')


    def list_parameters(self):
        """ a printable table of the current parameters and values"""
        return keyword_options.current_parameter_table(self)

