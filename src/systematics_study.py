"""Study systematics

"""
import os, sys
import numpy as np
import pylab as plt
import pandas as pd
import matplotlib.ticker as ticker

__docs__=['SystematicAnalysis']


from jupydoc import DocPublisher
# from lat_timing import Main

from utilities import (phase_plot , keyword_options)


class SystematicAnalysis(DocPublisher):
    """
    title: Fermi-LAT Exposure Systematics with Geminga
    author: T. Burnett <tburnett@uw.edu>
    sections:   data_set
                width_vs_cell_size
                adjacent_day_correlation
                correlation_vs_offset
                phase_plots
                summary

    source_name: Geminga
    """
    def __init__(self, source_name=None, **kwargs):
        super().__init__( **kwargs)
        self.source_name = source_name or self.source_name
  
    def data_set(self): 
        """Data Set

        Loaded **{self.source_name}** light curve  <a href="{link}">generated here.</a>: 
        {lc_df}

        Added *pull* distributions.
        """
        self.gdata = self.docman('GammaData.Geminga', as_client=True)
        link = self.docman.link
        self.gdata()
        self.light_curve = self.gdata.light_curve
        self.lc_df = lc_df = self.light_curve.dataframe
        lc_df['sigma'] = lc_df.errors.apply(lambda x: 0.5*(x[0]+x[1]))
        lc_df['pull'] = (lc_df.flux-1)/lc_df.sigma
    
        self.publishme()

    def introduction(self):
        """Introduction   
        
        This study uses, and for now, is specific to {self.source_name} data.
        We have 11 years, or ~4K independent daily measurments of the flux, which we can use to
        check the error estimates for systematics, especially the exposure. 
            
        - Dataset:  
        {self.gdata}    
        - 1-day light curve:  
        {self.lc_df}
        
        In what follows I examine the widths vs. interval size, and check for day-to-day correlations.
        """
        self.publishme()
        
    def width_vs_cell_size(self, nfit=6, interval_list=[1, 2, 4, 8, 16, 32, 50, 64, 90, 128, 256] ,):
        """Adjacent-day correlation
        
        Here I analyze the "pull", or normalized deviation of the flux measurement compared
        with the expected 1.0. The likelihood is of course Poissonian. The mean 
        number of counts per day is {mean_counts:.0f}; but the *effective* number of counts,
        determined from the shape of the likelihood curves, is {effective_counts:.0f}, is still
        large enough that the Gaussian approximation is fairly good. 
        I use the mean of the upper and lower errors for the sigma. 
        
        All values should be 1.0 if there are no systematics.
        {fig}
        
        The dashed line is a quadratic fit vs. $log(t)$ upto 30 days. Coefficients are {self.pfit_pars}.
  
        """
        lc = self.lc_df
        mean_counts = (lc.counts).mean()
        effective_counts = (lc.poiss_pars.apply(lambda x:x[1])).mean()
  
        # a list of light curves
        self.lcx = [self.light_curve]
        for i in interval_list[1:]:
            self.lcx.append(self.gdata.get_light_curve(i)) 
        yy = np.array([x.mean_std() for x in self.lcx])
        std = yy[:,1]
        class PolyFit(object):
            def __init__(self, xx, yy, order=2, ):       
                self.pars = np.polyfit(xx, yy, order, full=False)
                self.breakat=np.exp(xx.max())
                self.func = lambda x: np.poly1d(self.pars)(np.log(x))
                self.ymax = self.func(self.breakat)
            def __call__(self, xdom): 
                return np.where(xdom<self.breakat,self.func(xdom),[self.ymax])
    
        pfit = PolyFit(np.log(interval_list)[:nfit], std[:nfit]) 
        self.pfit_pars = np.array(pfit.pars).round(3)
        
        fig, ax = plt.subplots(figsize=(7,3), num=self.newfignum())
        ax.plot(interval_list,std, 'o');
        xdom = np.logspace(0,np.log10(100))
        ax.plot(xdom, pfit(xdom), '--', label='polynomial fit')
        ax.grid(alpha=0.5);
        ax.legend()
        ax.set(ylim=(1.0,None), xscale='log', xlabel='Cell size {days}', 
               ylabel='cell STD')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda val,pos: { 1.0:'1', 10.0:'10', 100.:'100'}.get(val,'')));

        self.publishme()
        
    def adjacent_day_correlation(self, lim=np.array([-4,4])):
        """        
        The single-day width is likely a systematic of the exposure. 
        If so, it should have a variation longer 
        than a day. To check this, examine the correlation of pulls for adjacent days.
        {fig}
        
        The correlation is not too evident in the scatter plot, but is significant, the
        correlation coefficient is {corr:.2f}!
        
        """
        pull = self.lc_df.pull.values
        x = pull[:-1]
        y = pull[1:]

        fig,ax=plt.subplots(figsize=(4,4),num=self.newfignum())
        ax.plot(x,y, '.');
        corr = np.sum(x*y)/(len(x)-1)
        ax.set(xlim=lim, ylim=lim, title='daily correlation of flux pull ', 
               xlabel="first day", ylabel="next day"); 
        ax.grid(alpha=0.5);
        ax.axhline(0,color='grey'); ax.axvline(0,color='grey')
        ax.plot(lim, corr*lim, '--', color='red', lw=2, label= f'correlation: {corr:.2f}');
        ax.legend(loc='upper left');
        
        self.publishme()
        
    def correlation_vs_offset(self, M:'Sample size for FFT'=2000):
        """Correlation vs. offset
        
        The adjacent day correlation, clearly indicates problems with the exposure. Here I look
        at the correlation coefficient vs. the offset. 
        {fig1}
        
        The peak at 1 year, and the precession period (55 days?) are apparent.
        
        Actually, a FFT reveals interesting structure:
        
        {fig2}
        The vertical dashed lines are at periods of 52 days and 26 days. There is quite a large peak 
        at 4/year, 91.3 day period.
                
        """
        lc = self.lc_df        
        plt.rc('font', size=14) ##??
        
        pull = lc.pull.values
        r = range(1,M)
        N = len(pull)
        corr = [np.sum(pull[:N-i]*pull[i:])/(N-i) for i in r]
        
        def correlation_plot(xmax=400):
            fig,ax = plt.subplots(figsize=(12,4), num = self.newfignum())
            ax.plot( r, corr, '-');
            ax.grid(alpha=0.5);
            ax.set(xlabel='offset [days]', xlim=(0,xmax), ylabel='correlation', 
                   title='Geminga flux 1-day correlation')
            ax.axhline(0, ls='-', color='grey');
            ax.axvline(365.25, ls=':', color='orange', label='1 year')
            ax.legend()
            return fig
        
        def fft_plot( ):
            yr = 365.25
            df = 1/M * yr
            output_array = np.fft.rfft(corr)
            power =np.square(np.absolute(output_array))# / norm
            fig,ax=plt.subplots(figsize=(12,3), num=self.newfignum())
            ax.plot(df*np.arange(len(power)), power, 'D', ms=7);
            ax.set(xlim=(0,0.05*yr), xlabel='Frequency (1/year)',
                  yscale='log', ylim=(1e-1,None), ylabel='Power',
                  xticks=np.linspace(0,20,11));
            ax.grid(alpha=0.75);
            for x in (yr/52., 2*yr/52):
                ax.axvline(x, ls='--', color='orange')
            return fig
        
        fig1=correlation_plot()
        fig2=fft_plot()
        self.publishme()
        
    def phase_plots(self):
        """Phase Analysis
        
        The discovery of long-tem periodicty in the correlations, especially a yearly 
        component, suggests that
        it could be corrected. Here I create phase plots to measure such a correction
        
        First try the precession period:
        {fig1}
        
        Not so dramatic: how about a quarter-yearly interval, the second-largest peak?
        {fig2}
        
        This should also show up in the yearly interval
        {fig3}
        Bingo! Obvious time-of-year dependence.
        
        Compare with the basic pull statistics:
        {fig4}
        The sigma, in relative flux, is {sigmean_pct:.1f}%. The systematic in the measured flux
        corresponds to {systematic_pct:.1f}%.
        So as yearly variation the systematic offset is of the order of the sigma, it may be
        comparable with the systematic broadenting.
        
        In fact, application of the yearly correction reduces the width from 
        {systematic_pct:.1f}% to 4.1%--details to come.
        
        But, as pointed out by Matthew, this could be related to fact that I don't have azimuthal corrections to the 
        PSF. 
        """
        df = self.lc_df # get 1-day light curve dataframe, with pulls calculated
        assert 'pull' in df, 'Expected pull to be calculated'
                
        fig1 = self.phase_plot(df,52.1)
        fig2 = self.phase_plot(df,365.25/4)
        fig3 = self.phase_plot(df,365.25)        
        fig4 = self.light_curve.fit_hists(fignum=self.newfignum())
        
        sigmean_pct= df.sigma.mean() * 100
        systematic_pct = (df.pull.std()-1)*100
        
        self.publishme()
        
    def phase_plot(self, df, period, bins=50, name='pull', ax=None):

        def phase_bins():
            phase = np.mod(df.t, period)
            phase_bin = np.digitize(phase, np.linspace(0,period,bins+1))
            g = df.groupby(phase_bin)
            p = g[name]
            return p.agg(np.mean), p.agg(np.std)/np.sqrt(p.agg(len))

        y, yerr = phase_bins()
        sig = (df.sigma).mean()
        y = 1+ y* sig
        yerr *= sig
        x = (np.arange(bins)+0.5)/bins
        fig,ax = (ax.figure, ax ) if ax else plt.subplots(figsize=(8,3), num=self.newfignum())
        ax.errorbar(x=x, y=y, yerr=yerr, fmt=' ', marker='o');
        ax.set(xlabel=f'phase within {period:.2f} day period', xlim=(0,1),
              ylabel='relative flux',)
        ax.axhline(1.0, color='grey')
        ax.grid(alpha=0.5)
        return fig
    
    def yearly_phase_plot(self,name='pull'):
        """Yearly phase
        
        The yearly phase plot for {name}.
        {fig}
        """
        df = self.lc_df # get 1-day light curve, with pulls calculated
 
        fig = self.phase_plot(df, period=365.25)
        
        self.publishme()
    
    def footer(self, source_file):
        # This is Fermi-LAT specific, assuming run at SLAC
        try:
            from __init__ import repository_name, SLAC_decorator, github_path
        except:
            return '*footer expects defining stuff in local __init__.py!*'
        code_link = SLAC_decorator(f'analysis/{repository_name}/{source_file}')
        if self.doc_folder:
            i = (self.doc_folder).find('burnett')+8
            doc_link = SLAC_decorator(self.doc_folder[i:]+'/index.html')
        else: doc_link=''
    
        r= self.markdown(
            f"""\
            ---
            This code, `{source_file}`, is part of my repository `{repository_name}`,
            and can be found at [github]({github_path}/{source_file})
            or, the current version (Fermi-LAT access) at [SLAC]({code_link}).

            (This document was created using [`jupydoc`](https://github.com/tburnett/jupydocdoc), 
            with Fermi-LAT web access [here]({doc_link}) )
            """
            )   
        return 
    
    def summary(self):
        """Summary and future plans
        
        I've found, and have a plan to correct for, a yearly variation of the flux.
        
        ### Plans:
        - Adjust measurements based on date  
          In progress: Effect is to reduce excess width from 7.8% to 4.1%  
          To do: apply corrections to Poisson fit when processing cells. Needs the -3 to 10% scale
          adjustment based on date, and a 4% broadening to account for other systematics.
          
        - Look for residual precession dependence  
          Done. No effect.
        - Try another bright pulsar
        - Apply to Bayesian Block studies
        """
        self.publishme()
    