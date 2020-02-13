"""
Code to apply the Atwood time-differencing algorithm
"""

import os, sys
import numpy as np
import pandas as pd
from scipy import random, optimize
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
        ('weight_key',      1,  '0: no weights; 1: use weights; -1: reverse'),
         
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
        
        window_size = window_size or self.window_size
        max_freq    =  max_freq   or self.max_freq
            
        time_data=self.dataset.time.values


        if self.use_weights:

            if self.weight_key==1:
                weights = self.dataset.weight.values
            elif self.weight_key==-1:
                weights = self.dataset.weight.apply(lambda x: 1-x).values
            elif self.weight_key==0:
                # all weights to 1
                weights = self.dataset.weight.apply( lambda x:  1 if np.isscalar(x) else len(x)).values
            else:
                raise Exception(f'Unexpected value for weight_key: {self.weight_key}')
        
            using_photons =  np.isscalar(weights[0])
            self.mean_weight =  weights.mean() if using_photons else sum(map(sum, weights))/sum(map(len, weights))
        else:
            self.mean_weight = 1.
        
        fft_size = 2 * int(np.floor(window_size * max_freq))
        
        if self.use_weights:
            weights = self.dataset.weight.values
            
        # generate time differences
        td = np.zeros(fft_size+2)
        for i1,t1 in enumerate(time_data):
            b = np.searchsorted(time_data, ( t1+window_size))
            t2 = time_data[i1+1:b]
            fb = np.floor((t2-t1)*max_freq*2).astype(int)
            
            td[fb] += weights[i1]*weights[i1+1:b] if self.use_weights else 1
#             td[fb] += weights[i1]*weights[i1+1:b] if using_photons else\
#                     [np.sum(np.outer(weights[i1],x)) for x in weights[i1+1:b]]
        self.diffs=td
        
        # run FFT
        norm = np.sum(np.absolute(td)/2.0, dtype=np.float32) 
        output_array = np.fft.rfft(td)
        power =np.square(np.absolute(output_array)) / norm

#         if self.make_plots:
            
#             if self.simulated:
#                 fig,(ax1,ax2) = plt.subplots(2,1, figsize=(10,8), gridspec_kw=dict(
#                     hspace=0.3,top=0.92,left=0.05)  )

#                 time_bin = window_size/fft_size
#                 times=np.linspace(0.5, fft_size-0.5, fft_size)*time_bin
#                 ax1.plot(times, td, '+');
#                 ax1.grid(alpha=0.5);
#                 ax1.set(xlabel='time difference [days]', ylim=(0,None), 
#                         ylabel='Entries per {:.1f} day'.format(time_bin))
#                 fig.suptitle('Processing: window size={}, max_freq={}'.format(
#                     window_size, max_freq),  ha='right')
#             else:
#                 fig, ax2 = plt.subplots(1,1, figsize=(8,3))
            
#                 deltaf= max_freq/fft_size*2
#                 freq = np.linspace(0.5, fft_size//2-0.5, fft_size//2)*deltaf
#                 ax2.plot( freq, power[1:], '+', color='grey', label='')
#                 ax2.grid(alpha=0.5);
#                 ax2.set(xlabel='Frequency [1/day]', ylabel='Power',yscale='log', **kwargs)
#                 ax2.text(0.65,0.8,
#                          f'photons     {len(time_data):8}\n'\
#                          f'window size {window_size:8}\n'\
#                          f'max_freq    {max_freq:8.1f}',
#                          fontdict=dict(family='monospace', size=10),
#                          transform=ax2.transAxes)
#                 if self.simulated:
#                     ax2.axvline(1/self.agn_period, color='red', ls=':', label='simulated frequency')
#                     ax2.legend()

#                 fig.set(facecolor='white')
        return power


    def list_parameters(self):
        """ a printable table of the current parameters and values"""
        return keyword_options.current_parameter_table(self)

    
class TDPower(object):
    def __init__(self, photon_data, window_size=400, max_freq=0.2, split=1, weight_key=1):
        
        self.window_size=window_size
        self.max_freq = max_freq
        df = photon_data
        if split>1:
            print(f'Splitting {len(df):,} photons into {split} data sets,', end='' )
        else:
            print ( f'Time differences for {len(df):,} photons', end='')
        print( f'  weight_key ={weight_key}')
        df8 = []
        td8 = []
        power = []
        tdiffs= []
        for i in range(split):
            if i>0: print(i, end='')
            df8.append( df[df.index%split==i])
            td8.append( TimeDiff(df8[i], window_size=window_size, 
                                    max_freq=max_freq, make_plots=False, weight_key=weight_key) )
            power.append(td8[i]())
        self.diffs = td8[0].diffs
        self.power = np.array(power); 
        self.photons = len(df)/split

        self.mean_weight = td8[0].mean_weight
        print(f'\npower spectrum shape: {self.power.shape}')
        
        t = self.pspec()
        self.x,self.y, self.yerr = t.x, t.y, t.yerr

    @classmethod
    def from_cells(cls, cdata, **kwargs):
        bw = cdata.data.binned_weights(None)
        cells=dict()
        for i,cell in enumerate(bw):
            cells[i]= dict(time=cell['t'], n=cell['n'], weight = cell['w'])
        df = pd.DataFrame.from_dict(cells, orient='index'); df.head()

        return cls(df, **kwargs)
        
    def pspec(self, offset=2, xmin=0.01):
        window_size = self.window_size; max_freq=self.max_freq
        fft_size = 2 * int(np.floor(window_size * max_freq))
        deltaf= max_freq/fft_size*2
        freq = np.linspace(0.5, fft_size//2-0.5, fft_size//2)*deltaf
        x = freq[offset:]
        p = self.power[:,offset+1:]
        y = p.mean(axis=0)
        yerr = p.std(axis=0)/3
        t= pd.DataFrame([x,y,yerr], 'x y yerr'.split()).T
        return t[t.x>xmin]


    def __call__(self,x):
        (a,b,c) = self.p
        return a*np.sqrt(np.power(x/1e-2,b)**2 + c**2)
#         e0=1e-2; gamma=0.5
#         (n0,cutoff,b) = self.p
#         return n0*(e0/e)**gamma*np.exp(-(e/cutoff)**b)
    
    def delta(self, p=None):
        if p is not None: 
            self.p=p
        return self.y / self(self.x) -1
    
    def fit(self, *pinit ):
        if len(pinit)==0:
            pinit = 0.25, -2, 100
        else:
            assert len(pinit)>=3
#         t = self.pspec()
#         self.x,self.y, self.yerr = t.x, t.y, t.yerr
        self.p=pinit
        fitfunc = lambda p: self.delta(p)
        pfit =  optimize.leastsq(fitfunc, self.p, ftol=1e-6,xtol=1e-6, maxfev=10000)[0]
        self.p=pfit
     
    def plot(self, ylim=(-0.1, 0.1), xlim=(1e-2, 0.2), title=None): 
        import matplotlib.ticker as ticker
        
        plot_fit = hasattr(self,'p')
        if plot_fit:
            fig, (ax1,ax2) = plt.subplots(2,1, figsize=(12,6), 
                                    gridspec_kw=dict(hspace=0.05,),sharex=True )
            dom = np.logspace(-2,-1)
            ax1.loglog(dom, self(dom), '--')
        else:
            fig, ax1 = plt.subplots(1,1, figsize=(12,4) )
            ax1.set( xscale='log', xlabel='Frequency (1/day)')
            ax1.xaxis.set_minor_formatter(ticker.FuncFormatter(
                lambda val,pos: {  2e-2:'0.02', 5e-2:'0.05', 2:'2', 5:'5', 20:'20', 50:'50'}.get(val,'')))                
            ax1.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda val,pos: { 1e-2:'0.01',  0.1:'0.1', 1:'1', 10:'10'}.get(val,'')))

        ax1.loglog(self.x, self.y, '.')
        ax1.set(xlim=xlim, ylabel='Power (arbitrary units)', ylim=ylim)
        ax1.grid(alpha=0.5, axis='x', which='both')
        ax1.grid(alpha=0.5, axis='y', which='major')
        ax1.text(0.75,0.65,
             f'photons     {self.photons:8.0f}\n'\
             f'window size {self.window_size:8}\n'\
             f'max_freq    {self.max_freq:8.1f}\n'\
             f'mean weight {self.mean_weight:8.2f} ',
             fontdict=dict(family='monospace', size=12),
             transform=ax1.transAxes)
        if title: ax1.set_title(title)
        if not plot_fit:
            return
        
        ax2.errorbar(self.x, self.delta(), yerr=self.yerr/self.y, fmt='+');
        ax2.set( xscale='log', xlabel='Frequency (1/day)',
               ylabel='relative deviation', ylim=ylim )
        ax2.xaxis.set_minor_formatter(ticker.FuncFormatter(
            lambda val,pos: {  2e-2:'0.02', 5e-2:'0.05', 2:'2', 5:'5', 20:'20', 50:'50'}.get(val,'')))                
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda val,pos: { 1e-2:'0.01',  0.1:'0.1', 1:'1', 10:'10'}.get(val,'')))
        ax2.grid(alpha=0.5, which='both');
        ax2.axhline(0, color='grey', ls='--')
        
        
#     def plot( self,  **kwargs) :

#         window_size = self.window_size; max_freq=self.max_freq
#         fft_size = 2 * int(np.floor(window_size * max_freq))
#         deltaf= max_freq/fft_size*2
#         freq = np.linspace(0.5, fft_size//2-0.5, fft_size//2)*deltaf
#         x = freq[1:]
#         p = self.power[:,2:]
#         y = p.mean(axis=0)
#         yerr = p.std(axis=0)/3
#         #print([t.shape for t in (x,p, y, yerr)])

#         fig, ax = plt.subplots(1,1, figsize=(12,6))

#         #ax.plot( freq[1:], power[2:], fmt='+', color='grey', label='')
#         ax.errorbar(x, y, yerr=yerr, fmt='.')
#         ax.grid(alpha=0.5);
#         ax.set(xlabel='Frequency [1/day]', ylabel='Power',yscale='log', **kwargs)
#         ax.text(0.65,0.8,
#                  f'photons     {self.photons:8.0f}\n'\
#                  f'window size {window_size:8}\n'\
#                  f'max_freq    {max_freq:8.1f}',
#                  fontdict=dict(family='monospace', size=12),
#                  transform=ax.transAxes)

#         fig.set(facecolor='white')

#     def polyfit(self, offset=4, order=5):
#         import matplotlib.ticker as ticker

#         t = self.pspec(offset)
#         x,y, yerr = t.x, t.y, t.yerr#self.pspec(offset)
#         ly = np.log(y)
#         kyerr = yerr/y
#         sx = 1/x
#         c = np.polyfit(sx, ly, order)

#         pfit = lambda x: np.polynomial.polynomial.polyval(1/x,c[::-1])
#         fig, (ax1,ax2) = plt.subplots(2,1, figsize=(12,8), sharex=True,
#                                       gridspec_kw=dict(hspace=0.05,))  
#         ax1.loglog(x, y, '+')
#         ax1.loglog(x, np.exp(pfit(x)), '-');
#         ax1.grid(alpha=0.5)
#         ax1.text(0.65,0.8,
#              f'photons     {self.photons:8.0f}\n'\
#              f'window size {self.window_size:8}\n'\
#              f'max_freq    {self.max_freq:8.1f}',
#              fontdict=dict(family='monospace', size=12),
#              transform=ax1.transAxes)

#         ax2.errorbar(x, y/np.exp(pfit(x))-1, yerr=kyerr, fmt='+');
#         ax2.set( xscale='log', xlabel='Frequency (1/day)', xlim=(1e-2,0.1),)# ylim=(-0.1,0.1))
#         ax2.xaxis.set_minor_formatter(ticker.FuncFormatter(
#             lambda val,pos: {  2e-2:'0.02', 5e-2:'0.05', }.get(val,'')))                
#         ax2.xaxis.set_major_formatter(ticker.FuncFormatter(
#             lambda val,pos: { 1e-2:'0.01',  0.1:'0.1'}.get(val,'')))
#         ax2.grid(alpha=0.5);
        