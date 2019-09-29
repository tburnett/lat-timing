"""
Develop alternate code to apply the Atwood time-differencing algorithm
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
        ('noise_rate'  ,0.2,   'noise photons/day'), 
        ('agn_period'  ,100.,  'QPO period in days'),  
        ('phase_width' , 0.05, 'spread in phase'),
        ('signal_count', 20,   'average number of counts per cycle'),

        'Analysis',
        ('window_size' , 400,  'window size in days'),
        ('max_freq'    , 0.1,  'maximum frequency day**-1'),

        'Processing options',
        ('make_plots',  True, 'generate plots for all steps'),
        ('binsize',     10,   'Binsize (in days) for time plots')
    )

    @keyword_options.decorate(defaults)
    def __init__(self, dataset=None, **kwargs):
        """

        """
        keyword_options.process(self, kwargs)
        self.dataset = dataset

        self.fft_size = 2 * int(np.floor(self.window_size * self.max_freq))
        self.time_resol = 0.5/self.max_freq

        if dataset is None:
            self.dataset=self.simulate()

        if 'weight' not in self.dataset.columns: self.use_weights=False
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
            fig,ax = plt.subplots(figsize=(10,3))
            hkw = dict(bins=np.linspace(0, self.viewing_period, nbins), histtype='step',lw=2)
            ax.hist(time_data, label='all photons', **hkw )
            ax.hist(signal,     label='signal', **hkw);
            ax.set(xlabel='Elapsed time [day]', ylabel='counts / {} days'.format(self.binsize))
            ax.legend();
            ax.set_title('Simulation: {} background and {} signal photons'.format(
                len(background),len(signal)))

        return pd.DataFrame([time_data], index=['time']).T


    def __call__(self, window_size=None):
        """ Run the time-differencing, then FFT
        """
        time_data=self.dataset.time.values
        if window_size is None: window_size=self.window_size
 
        td = np.zeros(self.fft_size)
        for i1,t1 in enumerate(time_data):
            b = np.searchsorted(time_data, ( t1+window_size))
            t2 = time_data[i1+1:b]
            fb = np.floor((t2-t1)/self.time_resol).astype(int)
            td[fb] += weights[i1]*weights[i1+1:b] if self.use_weights else 1
        
        norm = np.sum(np.absolute(td)/2.0, dtype=np.float32) 
        output_array = np.fft.rfft(td)
        self.power =np.square(np.absolute(output_array)) / norm

        if self.make_plots:
            fig,(ax1,ax2) = plt.subplots(2,1, figsize=(10,8), gridspec_kw=dict(hspace=0.3,top=0.95)  )
            ax1.plot(td, 'o');
            ax1.grid(alpha=0.5);
            ax1.set(xlabel='time difference bin', ylim=(0,None))

            ax2.plot(list(range(1, len(self.power))), self.power[1:], 'o');
            ax2.grid(alpha=0.5);
            ax2.set(xlabel='DFT bin', ylabel='Power')
            fig.suptitle('Processing: window size: {}, max_freq: {}'.format(window_size, self.max_freq))


    def list_parameters(self):
        """ a printable table of the current paramters and values"""
        return keyword_options.current_parameter_table(self)

