"""
"""
import os
import numpy as np
import pylab as plt
import pandas as pd

from jupydoc import DocPublisher

from lat_timing.light_curve import (PoissonRep, LightCurve, LightCurveX, LogLike, PoissonRepTable)
# from utilities import GammaData
github_code_path ='https://github.com/tburnett/lat_timing/tree/master/code'

__docs__= ['LATtiming']

class LATtiming(DocPublisher):
    """
    title: |
        LAT timing code
        Likelihood
        
    author: Toby Burnett        
    sections:
        title_page likelihood_definition        
    """
    #introduction likelihood_reps 
    def __init__(self,source_name='Geminga', **kwargs):
        super().__init__(**kwargs)
        self.gdata= GammaData(name=source_name)
        print('Loaded gamma data')
           
    def likelihood_definition(self, n=0, quiet=True):
        r"""Likelihood definition
        
        Here, from the Kerr derivation, is an approximation to the likelihood for
        a single cell, 
        \begin{align}
        \log\mathcal{L}(\alpha,\beta) =&\sum_{w} \log \bigg(1 + \alpha w + \beta\ (1-w)\bigg)
        - \alpha S - \beta B
        \end{align}
        
        where $w$ is a set of weights for the photons, $\alpha$ and $\beta$ are parameters
        representing the size of signal and background relative to their average, and
        $A$ and $B$ are estimates for the expected $\sum{w}$ and $\sum{(1-w)}$ from the 
        ensemble average. 
        """
        self.publishme()
    
    def test_data():
        """Test Data
        
        Loaded input data:  {gdata_str}
        Generate binned weights with `TimedData.binned_weights()`

        """
        gdata_str = self.monospace(str(self.gdata))
        t =self.gdata.cells.to_dict('records')
        bw = self.gdata.timedata.binned_weights()
        self.cells=bw
        self.cell =self.cells[n]
        self.publishme()
        
    def likelihood_reps(self):
        """Likelihood Representations
        
        Get a `LogLike` object with the selected cell.
        Explore how to represent the function, as defined by Kerr.
        
        The cell data: {ll.str}   
        
        * **Raw**
        The `LogLike` object implements the precise evaluation of the likelihood.         
        
        * **Gaussian**
        The `LogLike` object also performs a least-squares fit to itself. The parameters define the Gaussin 
        representation.
        Here is a plot of the least-squares result, compared with the likelihood function.
        {fig1}<br>       
        This shows the likelihood curve, and the fit result, with an error correspondong to the curvature.
        The point with an error bar is plotted at the -0.5 likelihood, where the upper and lower errors
        should be.
        
        * **Gaussian2D**
          The (source, backgrond) fit parameters.
          
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
        ll=LogLike(self.cell)
        counts = ll.n
        ll.str=self.monospace(repr(ll))
        fit_info=self.monospace(str(ll.fit_info()))
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