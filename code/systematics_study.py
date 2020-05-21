"""Study systematics

"""
import numpy as np
import pylab as plt
import matplotlib.ticker as ticker
import pandas as pd
import os, sys
import main, keyword_options
import docstring

class GammaData(main.Main):
    """## Geminga Data
    """

    defaults=main.Main.defaults+(
        ('radius', 7, 'ROI radius'),
        ('weight_file', '/nfs/farm/g/glast/u/burnett/analysis/lat_timing/data/weight_files', 'weight file dir'),
        ('source_name', 'Geminga', 'source name to use'), 
        ('interval', 1, 'days'),
        ('verbose', 0,  'verbosity'),

    )
    @keyword_options.decorate(defaults)
    def __init__(self, **kwargs):
        keyword_options.process(self,kwargs)        
        super().__init__( self.source_name,    **kwargs)

class SystematicAnalysis(docstring.Publisher):
    
    defaults=(
        ('source_name', 'Geminga', 'Name of the source to use'),
        ('interval_list', [1, 2, 4, 8, 16, 32, 50, 64, 90, 128, 256] ,'day intervals to try'),
        ('html_file', 'geminga_systematics.html', 'HTML output'),
        ('pdf_file', None, 'PDF output--not implemented yet'),
        ('gdata',    None, 'gamma data, a GammaData object--derive frpm source_name if not set'),
        ('no_display', False, 'flag to generate IPython display'),
        ('title', 'Fermi-LAT Exposure Systematics with source {source_name}', '' ),
        ('author', 'T. Burnett <tburnett@uw.edu>',''),
    )

    @keyword_options.decorate(defaults)
    def __init__(self,  **kwargs): 
        """
        """
        super().__init__()
        keyword_options.process(self,kwargs)
        self.title=self.title.format(source_name=self.source_name)
        
        # get the photon data, perhaps generating
        if self.gdata is None:
            self.gdata = GammaData(source_name=self.source_name)
        self.source_name = self.gdata.source_name
       
        #create a dataframe of the light curve based on 1-day fits, and add pull info
        self.lc = self.gdata.light_curve(1)

        cells= self.lc.dataframe
        wsum = [np.sum(c.w) for c in self.lc] # syn if weights
        cells['wfac'] = wsum/cells.counts
        cells['sigma'] = cells.errors.apply(lambda x: 0.5*(x[0]+x[1]))
        cells['pull'] = (cells.flux-1)/cells.sigma
        self.cells=cells

 

    def introduction(self):
    
        """
        
        This study uses, and for now, is specific to {self.source_name} data.
        We have 11 years, or ~4K independent daily measurments of the flux, which we can use to
        check the error estimates for systematics, especially the exposure. 
            
        - Dataset:  
        {self.gdata}    
        - 1-day light curve:  
        {self.lc}
        
        In what follows I examine the widths vs. interval size, and check for day-to-day correlations.
        """
        self.publishme('Introduction')
        
    def width_vs_cell_size(self, nfit=6):
        """
      
        
        Here I analyze the "pull", or normalized deviation of the flux measurement compared
        with the expected 1.0. The likelihood is of course Poissonian. The mean 
        number of counts per day is {mean_counts:.0f}; but the *effective* number of counts,
        determined from the shape of the likelihood curves, is {effective_counts:.0f}, is still
        large enough that the Gaussian approximation is fairly good. 
        I use the mean of the upper and lower errors for the sigma. 
        
        All values should be 1.0 if there are no systmatics.
        {fig}
        
        The dashed line is a quadratic fit vs. $log(t)$ upto 30 days. Coefficients are {self.pfit_pars}.
  
        """
        mean_counts = (self.cells.counts).mean()
        effective_counts = (self.cells.poiss_pars.apply(lambda x:x[1])).mean()
  
        # a list of light curves
        self.lcx = [self.lc]
        for i in self.interval_list[1:]:
            self.lcx.append(self.gdata.light_curve(i)) 
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
    
#         xx = np.log(self.interval_list[:nfit])     
        pfit = PolyFit(np.log(self.interval_list)[:nfit], std[:nfit]) 
        self.pfit_pars = np.array(pfit.pars).round(3)
        
        fig, ax = plt.subplots(figsize=(7,3), num=self.newfignum())
        ax.plot(self.interval_list,std, 'o');
        xdom = np.logspace(0,np.log10(100))
        ax.plot(xdom, pfit(xdom), '--', label='polynomial fit')
        ax.grid(alpha=0.5);
        ax.legend()
        ax.set(ylim=(1.0,None), xscale='log', xlabel='Cell size {days}', 
               ylabel='cell STD')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda val,pos: { 1.0:'1', 10.0:'10', 100.:'100'}.get(val,'')));

        self.publishme('The width of the pull vs cell size')
        
    def adjacent_day_correlation(self, lim=np.array([-4,4])):
        """
        
        The single-day width is likely a systematic of the exposure. 
        If so, it should have a variation longer 
        than a day. To check this, examine the correlation of pulls for adjacent days.
        {fig}
        
        The correlation is not too evident in the scatter plot, but is significant, the
        correlation coefficient is {corr:.2f}!
        
        """
        fig,ax=plt.subplots(figsize=(4,4),num=self.newfignum())
        pull = self.cells.pull.values
        
        x = pull[:-1]
        y = pull[1:]
        ax.plot(x,y, '.');
        corr = np.sum(x*y)/(len(x)-1)
        ax.set(xlim=lim, ylim=lim, title='daily correlation of flux pull ', 
               xlabel="first day", ylabel="next day"); 
        ax.grid(alpha=0.5);
        ax.axhline(0,color='grey'); ax.axvline(0,color='grey')
        ax.plot(lim, corr*lim, '--', color='red', lw=2, label= f'correlation: {corr:.2f}');
        ax.legend(loc='upper left');
        
        self.publishme('Adjacent-day correlation')
        
    def correlation_vs_offset(self, M:'Sample size for FFT'=2000):
        """
  
        The adjacent day correlation, clearly indicates problems with the exposure. Here I look
        at the correlation coefficient vs. the offset. 
        {fig1}
        
        The peak at 1 year, and the precession period (55 days?) are apparent.
        
        Actually, a FFT reveals interesting structure:
        
        {fig2}
        The vertical dashed lines are at periods of 52 days and 26 days. There is quite a large peak 
        at 4/year, 91.3 day period.
                
        """
                
        plt.rc('font', size=14) ##??
        
        pull = self.cells.pull.values
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
        self.publishme('Correlation vs. offset')
        
    def phase_plots(self):
        """

        The discovery of long-tem periodicty in the correlations, especially a yearly component, suggests that
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
        """
        df = self.lc.dataframe # get 1-day light curve, with pulls calculated
        assert 'pull' in df, 'Expected pull to be calculated'
                
        def phase_bins(period, bins=50, name='pull'):
            phase = np.mod(df.t, period)
            phase_bin = np.digitize(phase, np.linspace(0,period,bins+1))
            g = df.groupby(phase_bin)
            p = g[name]
            return p.agg(np.mean), p.agg(np.std)/np.sqrt(p.agg(len))
        
        def phase_plot(period, bins=50, name='pull'):
            y, yerr = phase_bins(period, bins, name)
            sig = (df.sigma).mean()
            y = 1+ y* sig
            yerr *= sig
            x = (np.arange(bins)+0.5)/bins
            fig,ax = plt.subplots(figsize=(8,3), num=self.newfignum())
            ax.errorbar(x=x, y=y, yerr=yerr, fmt=' ', marker='o');
            ax.set(xlabel=f'phase within {period:.2f} day period', xlim=(0,1),
                  ylabel='relative flux',)
            ax.axhline(1.0, color='grey')
            ax.grid(alpha=0.5)
            return fig
            
        fig1 = phase_plot(52.1)
        fig2 = phase_plot(365.25/4)
        fig3 = phase_plot(365.25)        
        fig4 = self.lc.fit_hists(fignum=self.newfignum())
        
        sigmean_pct= df.sigma.mean() * 100
        systematic_pct = (df.pull.std()-1)*100
        
        self.publishme('Phase Analysis')
        
        
    def footer(self, source_file):
        # This is Fermi-LAT specific, assuming run at SLAC
        try:
            from __init__ import repository_name, SLAC_path, github_path
        except:
            return '*footer expects defining stuff in local __init__.py!*'
        doc_path = '.' 
        if self.html_file:
            curpath = os.getcwd()
            rep_path, rep_name = os.path.split(SLAC_path)
            i = curpath.find(rep_name)
            j =curpath.find('/')+i
            if i==-1 or j==-1:
                doc_path = f'Problem construcing SLAC doc'
            else:
                rep_path+curpath[j-1:]+'/'+self.html_file+'?skipDecoration'
                doc_path= f', [{self.html_file} saved at SLAC (Fermi-LAT access)]({rep_path+curpath[j-1:]}/{self.html_file}?skipDecoration)'        
        r= self.markdown(
            f"""\
            ---
            This code, `{source_file}`, is part of my repository `{repository_name}`,
            and can be found at [github]({github_path}/{source_file})
            or, the current version (Fermi-LAT access) at [SLAC]({SLAC_path}/{source_file}?skipDecoration).

            (This document was created using [`docstring.py`]({github_path}/code/docstring.py) {doc_path}
            """
            )   
        return 
    
    def full_analysis(self, no_display=True):
        self.no_display=no_display
        self.header()
        self.introduction()
        self.width_vs_cell_size()
        self.adjacent_day_correlation()
        self.correlation_vs_offset()
        self.phase_plots()
        self.summary()
        self.footer('code/systematics_study.py')  
       
    def summary(self):
        """
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
        self.publishme('Summary and future plans')

    def __call__(self, selection=None, **kwargs):
        sections = [
             self.header,
             self.introduction,
             self.width_vs_cell_size,
             self.adjacent_day_correlation,
             self.correlation_vs_offset,
             self.phase_plots,
             self.summary,
             ]
        if selection is None:
            # all sections
            first=last=None
        if hasattr(selection, '__len__'):
            # range
            first, last = selection
        else:
            # single one
            first,last = selection, selection+1
        
        for section in sections[slice(first,last)]:
            print(f'running {section}')
            section()