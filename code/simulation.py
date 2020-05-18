"""
Manage simulation

"""

import os
import numpy as np
import pandas as pd
import pylab as plt

import keyword_options, light_curve
import docstring

class WeightGenerator(object):
    def __init__(self, weights, nbins=100, title='Weight generator'):

        self.nbins=nbins
        self.nw = len(weights)
        wbins = np.linspace(0,1,nbins+1)
        wh = np.histogram(weights, bins=wbins)[0]
        self.cumwt = np.cumsum(wh)/np.sum(wh)
        self.average = weights.mean()
        self.rms = np.sqrt(np.mean(weights**2))
        self.S = np.sum(weights)
        self.B = np.sum(1-weights)

        
    def doc(self, title='Weight generator'):
        """
        #### {title}
        
        Binned {self.nw} input weights into {self.nbins} bins.
        """       
        doc_display(WeightGenerator.doc)
        
    def __repl__(self):
        return f"""\
        \n#### {self.__class__.__name__}
        
        Binned {self.nw} input weights into {self.nbins} bins.
        """
    def __str__(self): return self.__repl__()
    
    def __call__(self, N):
        """return weights distributed according to input array
        N : float
            The expected number of weights: actual number will be Poisson-distributed
            
        
        """
        ngen = N if N>1000 else np.random.poisson(N,1)
            
        x = np.random.random(int(ngen)); 
        return np.searchsorted(self.cumwt, x)/self.nbins
    
    def test(self, title='Test generation'):
        """
        ### {title}
        Generate {self.nw} weights, compare mean, rms
        
        | quantity | input | generated  |
        | ---  | ---   | --- |
        | mean | {self.average:.3f} | {w_mean:.3f} |
        | rms  | {self.rms:.3f}    |  {w_rms:.3f}   |
        
        {fig}
        """
        w = self(self.nw)
        w_mean = w.mean()
        w_rms = np.sqrt(np.mean(w**2))
        fig,ax =plt.subplots(figsize=(5,3))
        ax.hist(w, np.linspace(0,1,51), histtype='step', lw=2)
        ax.set(xlabel='weight')
        fig.caption = 'Distribution of generated weights.'
        doc_display(WeightGenerator.test)

class SimulatedCells(list):
    """ Generate a list of cells
    
    """
    defaults =(
    ('width', 1, 'cell width'),
    ('fexp',  1,  'exposure per day'),
    ('verbose',0,  'verbosity'),
    )    
      
    @keyword_options.decorate(defaults)
    def __init__(self, ncells, average_counts, wtgen=None, **kwargs):
        """
        ncells : int
            number of cells to generate
        average_counts : float
            
        """
        keyword_options.process(self,kwargs)
        verbose=self.verbose
        class SimData(object):
            def __init__(self):
                self.source_name='simulation'
                self.edges=None
                self.verbose = verbose
        self.data = SimData()
        if wtgen:
            wtmean = wtgen.average
            self.S = wtgen.S
            self.B = wtgen.B
            N = wtgen.nw
        else:
            wtgen = lambda n: np.ones(n)
            wtmean = 1.
            N = ncells * average_counts
            self.S,self.B = N,0
        
        # generate counts per cell
        counts = np.random.poisson(lam=average_counts, size=ncells)
         
        for i,n in enumerate(counts):
            w = wtgen(n)
            #exp = n*self.fexp/N
            exp = average_counts*self.fexp/N
            self.append(dict(
                t=(i+0.5)*self.width,
                tw=self.width,
                fexp=self.fexp,
                n=n,
                w =w,
                S=exp * self.S,
                B=exp * self.B,            
            ))

class Simulation(docstring.Displayer):
    
    def __init__(self, ncells=4000, average_counts=50, wtgen=None ):
        """### Simulation setup
            
            {self.__class__.__name__} invoked to enerate {ncells} cells, {average_counts} per cell.  
            {weight_text}
            <br>Fits likelihood to Poisson-like function. 
              
        """
        super().__init__()
        # make a set of cells
        self.cells = SimulatedCells(ncells, average_counts, wtgen)
        
        weight_text = f'Generating weights, mean = {wtgen.average:.2f}' if wtgen else 'no weights' 
        # feed to light_curve
        self.lc = light_curve.LightCurve(self.cells)
        
        self.display()
        
class SimulationFromData(docstring.Displayer, list):
    
    def __init__(self, cdata, verbose=0, interval=1):
        """#### Simulation from {cdata.name} data
                
        * Weight generation with {wgen.__class__.__name__}:
        Average number of weights per cell: {lfac:.0f}
        """
        super().__init__()
        class SimData(object):
            def __init__(self):
                self.__dict__.update(dict(
                    source_name=f'{cdata.name} simulation',
                    edges=None,
                    verbose=verbose,
                    ))
        self.data = SimData()
        
        # setup weight generator from data weights
        data_w = cdata.photons.weight
        wgen = WeightGenerator(data_w)
        self.S = np.sum(data_w)
        self.B = np.sum(1-data_w)
        bw = cdata.data.binned_weights(None)
        N = len(bw)
        lfac = (bw.S+bw.B)/N

        for i,cell in enumerate(bw):
            fexp = cell['fexp']
            exp = fexp/N
            w = wgen(fexp*lfac)
            self.append(dict( 
                             t = cell['t'],
                             tw= interval,
                             fexp=fexp,
                             n=len(w),
                             w=w,
                             S=exp * self.S,
                             B=exp * self.B,
                            ))        
        
        self.display()
        
    
class LightCurveDisplay(docstring.Displayer):
    """ Special class to display light curve info"""
    
    def __init__(self, light_curve, title='## Light curve', path=None):
        """
        {title}
        """
        super().__init__(path=path)        
        self.lc = light_curve
        self.display()
        
    def exposure(self):
        """
        #### Exposure  
        Exposure per day, relative to average.
        {fig}
        """
        fexp=self.lc.dataframe.fexp.values
        if fexp.std()<0.01:
            fig = f'Constant: std={fexp.std():.2f}<0.01'
        else:
            fig,ax = plt.subplots(figsize=(4,2), num=self.newfignum())
            ax.hist(fexp, 100)
            ax.grid(alpha=0.5)
            ax.set(xlabel='exposure factor')
        self.display()
        
    def summary_plots(self):
        """
        #### Flux per day 
        {fig1}

        ### rate
        {fig2}

        """
        fig1,ax1 = plt.subplots(figsize=(12,4),  num=self.newfignum())
        fig1.caption = """Measured flux per day"""
        self.lc.flux_plot(ax=ax1,title=None)
        
        
        self.lc.fit_hists(fignum=self.newfignum())
        fig2=plt.gcf()
        fig2.caption='The distributions of the Poisson fit results.'
        
        self.display()
        
    def statistics_summary(self, title='Cell Data'):
        """
        #### {title}

        * **Head of the cell fit data frame**
        {dfhead}
        The "poiss_x" columns are the three poisson-like parameters:
        "sp", "e", and "b" are peak value, equivalent counts, and background, respectively.

        * **Statistics**
        {stats}

        """
        df = self.lc.dataframe.copy()

        stuff = [ 'mean', 'std', 'min', 'max',]
        for i,par in enumerate('sp e b'.split()):
            df[f'poiss_{par}'] = df.poiss_pars.apply(lambda x: x[i])
        del df['poiss_pars']
        dfhead = df.head(2)
        stats =df['counts fexp flux ts  poiss_sp poiss_b poiss_e'.split()].agg(stuff).T 

        self.display()

    def all_plots(self):
        self.exposure()
        self.statistics_summary()
        self.summary_plots()
