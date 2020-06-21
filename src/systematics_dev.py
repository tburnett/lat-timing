"""
package dev initialization
"""
import os, sys
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

__docs__=['Development']

from jupydoc import DocPublisher

from utilities import phase_plot, poiss_pars_hist, GammaData
#from utilities import PoissonTable
   
class Development(DocPublisher): 
    """
    title: Systematics Code Development
    author: THB
    
    sections:
        title_page
        setup_summary
        yearly_systematic
        poisson_parameters
        manipulate_poisson
         
    """
    
    def __init__(self, gamma_data=None, source_name=None, **kwargs):
        super().__init__( **kwargs)
        if not gamma_data:
            assert source_name,'Expect source_name is set'
            gamma_data= GammaData(name=source_name)
        self.gdata = gamma_data
        self.source_name = gamma_data.source_name
        
        # get 1-day poisson fit light curve, and save mean sigma, pull
        self.lc = gamma_data.light_curve()
        lcdf = self.lc.dataframe.copy()
        
        # make dataframe from it with Gaussian-approximation pulls
        df= lcdf.loc[:,'t counts fexp flux'.split()]
        df['sigma'] = lcdf.errors.apply(lambda x: 0.5*(x[0]+x[1]))
        df['pull'] = (df.flux-1)/df.sigma
        self.df = df
      
    def setup_summary(self, display=True): 
        """Setup summary
        
        Loaded gamma data:
        
        {gdata_summary}
        
        From the 1=day light-curve DataFrame, create new one including $t, counts, fexp, flux$,
        with $sigma, pull$ added.
        {dfhead}

        """        
        gdata_summary = self.monospace(self.gdata)
        dfhead = self.df.head(1)
        #---------------------------
        self.publishme()
        
    def yearly_systematic(self, display=True, period =365.25, nbins=50,):
        r"""
        The daily pull distribution is too wide: we saw that it is {pull_std:.3f}.
        The  likelihood normalization yearly dependence may contribute:
        {figa}
        
        
        So let's correct each flux measurement: Given that the pull $p$ is defined in terms of
        the flux $f$ and uncertainty $\sigma$ by $p=(f-1)/\sigma$, then adjusting the flux measument 
        implies $f^{\prime} = 1 + \sigma(p + \Delta p)$.
        
        After adustment, we get {adjusted_pull_std:.3f}.
        {figb}
        {figb.number}
        The effect on the distribution is obviously not large viewed this way.
        But, as we will see, the effect on multple days is larger.
        
        """
        #-----------------------------------------------------
        df = self.df # get 1-day light curve, with pulls calculated
        pull_std = df.pull.std()
        
        figa= phase_plot(df, period=period)
        plt.gcf().numeber = self.newfignum()
  

        def pull_systematic(df, period, nbins=50):
            """ analyze the dataframe: assume has t, pull
            set df.pull_adjusted
            """
            # yearly phase

            phase = np.mod(df.t, period)
            x_phase = (np.arange(nbins)+0.5)/nbins
            phase_bin = np.digitize(phase, np.linspace(0,period,nbins+1))-1
            # make a group with the phase bins
            g=phase_group = df.groupby(phase_bin)
            g_pull_mean = g['pull'].agg(np.mean)
            return dict(period=period, nbins=nbins,
                       correction=g_pull_mean)
        
        self.flux_fix = fix = pull_systematic(df, period, nbins)
        
        def pull_adjustment(df, fix):
            # Apply back to the pulls 
            phase = np.mod(df.t, period)
            phase_bin = np.digitize(phase, np.linspace(0,period,nbins+1))-1
            pull_mean = fix['correction']

            # make a list of the values for each cell
            corr = np.array([pull_mean[i] for i in phase_bin]);
            # adjust the pull
            return  df.pull.values - corr    
        
        adjusted_pull = pull_adjustment(self.df, fix)
          
        adjusted_pull_std = adjusted_pull.std()
        
        def pulls_hist():
            fig, ax = plt.subplots(figsize=(6,3), num=self.newfignum())
            hkw = dict(bins = np.linspace(-5,5, 51),histtype='step', lw=2, log=True,)
            for x, name in zip([self.df.pull, adjusted_pull], 'raw adjusted'.split()):
                ax.hist(x, label=f'{name:8s} {x.std():.2f}', **hkw )
            ax.set(xlabel='normalized deviation, or pull',ylim=(0.5,None))
            leg=ax.legend(prop=dict(family='monospace')); 
            ax.grid(alpha=0.5)
            ax.axvline(0,color='grey')
    
            return fig
        
        figb = pulls_hist()
        figb.caption=f'Figure {figb.number}'
            
  
        #------------------------------------------------------------  
        self.publishme('Determine yearly systematic')

    def poisson_parameters(self):
        """Poisson parameter distribution
        
        #### The distributions of the poisson        
        This data set, with source {self.source_name}.
        
        Distributions of the three Poisson parameters
        {fig1}
        
        ####  Modification
        Construct the likelihood function, as represented by a Poisson instance
        and modify its parameters
        
        Set the parameters {ppars}, and try another, with flux reduced by 0.5, add counts divided by
        4, to increase width by factor of 2.
        {fig2}
        """
        from lat_timing.poisson import Poisson, PoissonFitter
        
        pars = self.lc.dataframe.poiss_pars
        
        fig1 = poiss_pars_hist(pars, fignum=self.newfignum())
        fig1.caption = f'Fig {fig1.number}. The Poisson parameters for source {self.source_name}.'
        #-------------- 
        ppars = (1.0, 250, 0.2)

        p =Poisson(ppars);
        px = Poisson([ppars[0]/2, ppars[1]/4, ppars[2]])
        def figa():
            xlim = (0.,1.2)
            x = np.linspace(*xlim)
            fig, ax = plt.subplots(figsize=(4,2), num=self.newfignum())
            ax.plot(x, p(x), label='original')
            ax.plot(x, px(x), label='adjusted')
            ax.set(ylim=(-4, 0.5))
            ax.grid(alpha=0.5);
            ax.axhline(0, color='grey', ls='--')
            ax.axvline(1.0, color='grey', ls='--');
            ax.legend()
            return fig
        fig2 = figa()

        #-------------------------------------------
        self.publishme()

        
    def manipulate_poisson(self):
        """Poisson manipulation
        
        #### PoissonTable class
        We need a way to efficiently combine poisson-like likelihoods. For this
        I have a new class "PoissonTable", which can be instantiated from the poisson parameters,
        or from a table generated by another such.
        
        For Poisson parameters {ppars}, corresponding to daily Geminga data,  here it is:
        {figb}
        the range correspond to limits of $10^{{-6}}$ in integrated probability on both ends.
        
        #### Add another poisson
        We will add the likelilhood table to combine with another likelihood function
        {figc}
        
        """        
        #-------------------------------------
        ppars = (1.0, 200, 0.2)   
        pt=PoissonTable.from_poisson(ppars)
        figb = pt.plot(fignum=self.newfignum())
        self.add_caption(' Initial likelihood')

        pt.add(ppars)
        figc = pt.plot(fignum=self.newfignum())
        self.add_caption(' After adding another one')
        
        #-------------------------------------
        self.publishme()

#     def add_caption(self, text):
#         fig = plt.gcf()
#         fig.caption=f'Fig. {fig.number}. {text}'
        
