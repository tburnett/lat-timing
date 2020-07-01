"""
"""
import os
import numpy as np
import pylab as plt
import pandas as pd

from jupydoc import DocPublisher

from lat_timing.light_curve import (PoissonRep, GaussianRep, Gaussian2dRep, LogLike)#, PoissonRepTable)

github_code_path ='https://github.com/tburnett/lat_timing/tree/master/code'

__docs__= ['LikelihoodReps']

class LikelihoodReps(DocPublisher):
    """
    title: Likelihood Representations
        
    author: Toby Burnett        
    sections:
        data_set 
        likelihood_reps 

    source_name: Geminga    
    """
    #introduction likelihood_reps 
    def __init__(self,source_name=None, **kwargs):
        super().__init__(**kwargs)
        self.source_name = source_name or self.source_name

    def data_set(self): 
        """Data Set

        Loaded **{self.source_name}** photon data <a href={link}>generated here.</a>: 
        {lc_df}

        Added *pull* distributions.
        """
        
        self.gdata, link = self.docman.client('GammaData.Geminga')
        self.light_curve = self.gdata.light_curve
        self.lc_df = lc_df = self.light_curve.dataframe
        lc_df['sigma'] = lc_df.errors.apply(lambda x: 0.5*(x[0]+x[1]))
        lc_df['pull'] = (lc_df.flux-1)/lc_df.sigma
    
        self.publishme()
    
       
    def likelihood_reps(self):
        """Likelihood Representations
        
        Get a `LogLike` object with the selected cell.
        Explore how to represent the function, as defined by Kerr.
        
        The cell data: {ll.str}   
        
        * **Raw**
        The `LogLike` object implements the precise evaluation of the likelihood. 
                
        
        * **Gaussian**
        The `LogLike` object also performs a least-squares fit to itself. 
        The parameters define the Gaussian representation.
        Here is a plot of the least-squares result, compared with the likelihood function.
        {fig1}<br>       
        This shows the likelihood curve, and the fit result, with an error correspondong to the curvature.
        The point with an error bar is plotted at the -0.5 likelihood, where the upper and lower errors
        should be.

        {grep}
        
        * **Gaussian2D**
          The (source, backgrond) fit parameters.
          
          {g2rep}
          
        * **Poisson**
        The LogLike object is a likelihood function. But it is a bit time consuming, involving sums over the 
        weights, {counts} in this case. On the other hand, a Gaussian constructed from the least-squares fit
        is, for high statistics as this case, is a very good approximation.
        <br>I have found that a Poisson-like function works quite well in all cases, and I use it extensively.
        The code, in [poisson.py]({github_code_path}/poisson.py), is invoked by `PoissonRep`. It
        is much faster, 5.45 µs vs. 77.4 µs in this case, and requires only three parameters vs the entire
        array of weights.
        <br>Fit results: {prep}
        <br> Comparison plot.
        {fig2}
        
        * **Poisson Table**
        The Bayesian Block algorithm requires multiple evaluations of the log-likelihood, adding the
        curves. The table lookup evaluation at a single point is no faster, but the tables are available, so this operation
        is simply adding arrays to get a new function. Thus this representaion allow a fast addition of two
        log-likelihoods, following the overhead to create the table.
  
        """
        self.cell = self.gdata.cells.iloc[0]
        ll = LogLike(self.cell)
        counts = ll.n
        ll.str=self.monospace(repr(ll))
        fit_info=self.monospace(str(ll.fit_info()))

        grep = GaussianRep(ll).fit
        g2rep= Gaussian2dRep(ll).fit
        prep = PoissonRep(ll).fit

        def gaussian_fit():
            fig, ax = plt.subplots(figsize=(4,2))
            ll.plot(ax=ax)
            ax.set_title(f'Fit to data for single cell')
            return fig
        fig1 = gaussian_fit()
        
        prep = PoissonRep(ll)
        def poisson_fit():
            fig, ax = plt.subplots(figsize=(6,3))
            prep.comparison_plots(ax=ax)
            return fig
        fig2 = poisson_fit()
        
        self.publishme()