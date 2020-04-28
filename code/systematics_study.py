"""Study systematics

"""
import numpy as np
import pylab as plt
import matplotlib.ticker as ticker
import pandas as pd
import os, sys
import main, keyword_options

class GemingaStudy(object):
    """ # Geminga Study
    """

    defaults=(
        ('radius', 7, 'ROI radius'),
        ('weight_file', '/nfs/farm/g/glast/u/burnett/analysis/lat_timing/data/weight_files', 'weight file dir'),
        ('source_name', 'Geminga', 'source name to use'), 
        ('verbose', 0,  'verbosity'),
    )
    @keyword_options.decorate(defaults)
    def __init__(self, interval, **kwargs):
        keyword_options.process(self,kwargs)
        print(f'source {self.source_name}: interval {interval} day(s)')
        self.cdata = main.Main( self.source_name, interval=interval, 
                                weight_file=self.weight_file,
                                data_selection=dict(radius=self.radius),
                                verbose=self.verbose, 
                                **kwargs)
        self.lc = self.cdata.light_curve()

    def rms(self):
        df = self.lc.fit_df
        frms = df.errors.apply(lambda err: 0.5*(err[0]+err[1]))
        fwts = 1/frms**2
        fmean = np.sum(df.flux*fwts)/np.sum(fwts)
        t = (df.flux-fmean)/frms
        return t.std()

    def flux_plot(self, **kwargs):
        self.lc.flux_plot(**kwargs)
        
    def residual_plot(self, title=None):
        df = self.lc.fit_df
        frms = df.errors.apply(lambda err: 0.5*(err[0]+err[1]))
        fwts = 1/frms**2
        fmean = np.sum(df.flux*fwts)/np.sum(fwts)
        t = (df.flux-fmean)/frms
        self.wrms = np.sqrt(1/np.sum(fwts))
        #print(f'Total sigma: {wrms:.3e}')

        def fig1(ax):
            low = df.fexp<0.7
            high = df.fexp>1.4
            med = np.logical_not(np.logical_or(low,high))
            hkw = dict(bins= np.linspace(-5,5,41), histtype='step', lw=2, log=True)
            for cut,label in zip([low,med,high], 'low med high'.split()):
                u =t[cut]
                ax.hist(u, label=f'{label:3} {u.mean():.2f} {u.std():.2f}', **hkw)

            ax.set(xlabel='deviation')
            ax.legend()

        def fig2(ax):
            ax.plot(df.fexp, t, '.')
            ax.set(xlabel='exposure',xscale='log', ylabel='deviation', ylim=(-4,4))
            ax.axhline(0,)
            xticks = [0.5, 1.0, 2.0]
            ax.set_xticks(xticks)
            ax.set_xticklabels(map(lambda x:f'{x:0.1f}', xticks))

        fig, axx = plt.subplots(1,2, figsize=(12,5))
        for ax,figure in zip(axx.flatten(), [fig1,fig2]):
            figure(ax)
            ax.grid(alpha=0.5)
        if title is not None:
            fig.suptitle(title)
        
        
class MultipleIntervals(list):
    """ just run multiple intervals"""

    def __init__(self, interval_list):
        self.interval_list = interval_list
        for i in interval_list:
            self.append( GemingaStudy(interval=i) )
        self.widths = [s.rms() for s in self]

class Analyze(list):
    def doc(self):
    """ * class Analyze: 
    make plots from a set of intervals
    """
    
    
    def __init__(self, multi):
        self.interval_list=multi.interval_list
        self+=multi
        self.widths = [s.rms() for s in self]
        
    def plot_values(self, ax=None):
        ax = ax or plt.gca()
        ax.plot(self.interval_list, self.widths, 'o--');
        ax.set( ylim=(1,None), ylabel='Cell STD', xlabel='cell size (days)', xscale='log')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda val,pos: { 1.0:'1', 10.0:'10', 100.:'100'}.get(val,'')));
        ax.grid(alpha=0.5)
       
    def logpolyfit(self, nfit=6, order=2):

        xx = np.log(self.interval_list[:nfit]) 
        yy = self.widths[:nfit]
        self.pf = np.polyfit(xx, yy, order, full=False)
        print(f'Log of order {order} polynomial fit parameters to'\
              f' {self.interval_list[nfit]} days: {self.pf.round(4)}')
     
    def plot_fit(self, breakat=30):
        yfit = np.poly1d(self.pf)
        xdom = np.logspace(0,np.log10(128))
        ax=plt.gca()
        func = lambda x: yfit(np.log(x))
        ax.plot(xdom, np.where(xdom<breakat,func(xdom),[func(breakat)]), '--',
               label='polynomial fit');
        ax.plot(self.interval_list,self.widths, 'o', label='measurements');
        ax.set(xscale='log',  ylim=(1,None),
                ylabel='Systematics factor', xlabel='cell size (days)');
        ax.grid(alpha=0.5);
        ax.legend()
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda val,pos: { 1.0:'1', 10.0:'10', 100.:'100'}.get(val,'')));