"""
"""


import os, sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from lat_timing import Poisson, PoissonFitter

from jupydoc import DocPublisher

__docs__ =['PoissonDoc',]

class PoissonDoc(DocPublisher):
    """
    title: Poisson fitting 
   
    """
    def title_page(self):
        r"""<h2>{title}</h2>

        The module `lat_timing.poisson` has two classes:
        * `PoissonFitter`&mdash;fits a function to the Poisson-like function 
        * `Poisson` &mdash;Poisson-like function

        The function, as described  in Nolan et al., 2003, ApJ 597:815:627, is
        
        $log\mathcal{L}(s) = e\ (s_p+b) \log \big(\ e\ (s+b) \big) -e\ s $

        where $s$ is the flux, $s_p$ the value of $s$ at the peak (which may be the 
        limit zero), $b$ the background, required to be >0, and $e$ a normalization 
        factor to convert flux to equivalent counts.  It has been observed to be a good
        representation of the likelihood in gamma-ray astronomy.

        A quick demonstration is to use `Poisson` to define such a function, and see if `PoissonFitter` returns the same parameters 
        Here are two examples, first a function peaking at -10, so only a limit, and a
        second  peaking at 50. Both have backgrounds of 1.0.
        {fig}


        """
        title = self.doc_info['title']
        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(6,3)) 
        pf = PoissonFitter((Poisson([-10, 1., 10.])))
        #print(np.array(pf.fit()).round(3))
        pf.plot(ax=ax1)
        self.pf = pf

        pf2 = PoissonFitter( Poisson([50., 1., 10.]))
        self.pf2 = pf2
        #print(np.array(pf2.fit()).round(3))
        pf2.plot(ax=ax2)

        #--------------------------
        self.publishme()

    

