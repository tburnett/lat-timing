"""Manage the binned time data to produce light curves, etc.

"""
import numpy as np
import pylab as plt
import pandas as pd
import os, sys

from scipy import (optimize, linalg)
from scipy.linalg import (LinAlgError, LinAlgWarning)
import keyword_options, poisson
from astropy.stats.bayesian_blocks import * 

data=None # data_managment.Data obect for access to name, verbose, etc.

    
class LogLike(object):
    """ implement Kerr Eqn 2 for a single interval, or cell"""
    
    def __init__(self, cell):
        """ cell is a dict"""       
        self.__dict__.update(cell)
        
    def fit_info(self, fix_beta=True):
        """Perform fits, return a dict with cell info"""
        pars = self.solve(fix_beta)
        if pars is None:
            if data.verbose>0:
                print(f'Fail fit for {self}')
            return None
            #raise RuntimeError('Fit failure')
        hess = self.hessian(pars)
        outdict = dict(t=self.t, tw=self.ts, fexp=self.fexp, counts=len(self.w) )
        if len(pars)==1:
            outdict.update(flux=pars[0], sig_flux=np.sqrt(1/hess[0]))
        else:
            beta = pars[1]
            var  = np.linalg.inv(hess)
            err  = np.sqrt(var.diagonal())
            sig_flux=err[0]
            sig_beta=err[1]
            corr = var[0,1]/(err[0]*err[1])
            outdict.update(flux=pars[0], beta=beta, 
                        sig_flux=sig_flux, sig_beta=sig_beta,corr=corr)
        return outdict
        
    def __call__(self, pars ):
        """ evaluate the log likelihood 

        """
        pars = np.atleast_1d(pars)
        if len(pars)>1:      alpha, beta = pars
        else:                alpha, beta = max(-1, pars[0]-1), 0
            
        return np.sum( np.log(1 + alpha*self.w + beta*(1-self.w) )) - alpha*self.S - beta*self.B

    def __repr__(self):
        return f'{self.__class__.__module__}.{self.__class__.__name__}:'\
        f' time {self.t:.3f}, {len(self.w)} weights,  exposure {self.fexp:.2f}, S {self.S:.0f}, B {self.B:.0f}'
        

    def gradient(self, pars ):
        """gradient of the log likelihood with respect to alpha=flux-1 and beta, or just alpha
        """
        w,S = self.w, self.S
        pars = np.atleast_1d(pars)
  
        alpha =  max(-1,pars[0] -1)        
        if len(pars)==1:           
            D = 1 + alpha*w
            return np.sum(w/D) - S
        else:
            beta = pars[1]
            D =  1 + alpha*w + beta*(1-w)
            da = np.sum(w/D) - S
            db = np.sum((1-w)/D) - self.B
            return [da,db]  
        
    def hessian(self, pars):
        """return Hessian matrix (1 or 2 D according to pars) from explicit second derivatives
        Note this is also the Jacobian of the gradient.
        """
        w = self.w
        pars = np.atleast_1d(pars)
        alpha = max(-1, pars[0]-1)
        if  len(pars)==1:
            D = 1 + alpha*w 
            return [np.sum((w/D)**2)]
        else:
            beta= pars[1]
            Dsq = (1 + alpha*w + beta*(1-w))**2
            a, b, c = np.sum(w**2/Dsq), np.sum(w*(1-w)/Dsq), np.sum((1-w)**2/Dsq)
            return np.array([[a,b], [b,c]])
        
    def rate(self, fix_beta=True, debug=False, no_ts=True):
        """Return signal rate and its error"""
        try:
            s = self.solve(fix_beta )
            if s is None:
                return None
            h = self.hessian(s)
        
            v = 1./h[0] if fix_beta else linalg.inv(h)[0,0]
            ts = None if no_ts else (0 if s[0]<=-1 else 2*(self(s)-self([-1,s[1]])))
            return (s[0]), np.sqrt(v), ts
        
        except (LinAlgError, LinAlgWarning, RuntimeWarning) as msg:
            if debug or data.verbose>2:
                print(f'Fit error, cell {self},\n\t{msg}')
        except Exception as msg:
            print(f'exception: {msg}')
        print(9999.)
            
    def minimize(self,   fix_beta=True,estimate=[0.5,0], **fmin_kw):
        """Minimize the -Log likelihood """
        kw = dict(disp=False)
        kw.update(**fmin_kw)
        f = lambda pars: -self(pars)
        return optimize.fmin_cg(f, estimate[0:1] if fix_beta else estimate, **kw)

    def solve(self, fix_beta=True, debug=False, estimate=[0.5,0],**fit_kw):
        """Solve non-linear equation(s) from setting gradient to zero 
        note that the hessian is a jacobian
        """
        kw = dict(factor=2, xtol=1e-3, fprime=self.hessian)
        kw.update(**fit_kw)
        if fix_beta and self.gradient([0])<0:
            # can happen if solution is negatives
            return [0]
        try:
            ret = optimize.fsolve(self.gradient, estimate[0] if fix_beta else estimate , **kw)   
        except RuntimeWarning as msg:
            if debug or data.verbose>2:
                print(f'Runtime fsolve warning for cell {self}, \n\t {msg}')
            return None
        return np.array(ret)
        
    def plot(self, fix_beta=True, xlim=(0,1.2),ax=None, title=None):
        fig, ax = plt.subplots(figsize=(4,2)) if ax is None else (ax.figure, ax)

        dom = np.linspace(*xlim)
        a, s, ts = self.rate(fix_beta=fix_beta, debug=True)
        if fix_beta:
            f = lambda x: self([x])
            beta=0
        else:
            a, beta = self.solve(fix_beta, debug=True)
            #assert abs(a-v+1)<1e-6, f'{a},{v}, {a-v+1}'
            f = lambda x: self([x, beta])
        ax.plot(dom, list(map(f,dom)) )

        ax.plot(a, f(a), 'or')
        ax.plot([a-s, a+s], [f(a-s), f(a+s)], '-k',lw=2)
        for x in (a-s,a+s):
            ax.plot([x,x], [f(x)-0.1, f(x)+0.1], '-k',lw=2)
        ax.plot(a, f(a)-0.5, '-ok', ms=10)
        ax.grid()
        ax.set(title=title, xlim=xlim, ylim=(f(a)-4, f(a)+0.2), 
               ylabel='log likelihood', xlabel='flux')


class GaussianRep(object):
    """ Manage fits to the loglike object
    """
    
    def __init__(self, loglike, fix_beta=True):
        """1- or 2-D fits to LogLike"""
        self.fix_beta = fix_beta
        self.fit = loglike.fit_info(fix_beta)        
        
    def __call__(self, pars):
        return None # TODO if needed

    def __repr__(self):
        return f'{self.__class__.__module__}.{self.__class__.__name__}: {self.fit}'
        
class Gaussian2dRep(GaussianRep):
    def __init__(self, loglike):
        super(Gaussian2dRep, self).__init__(loglike, fix_beta=False)
    
        
class PoissonRep(object):
    """Manage the representation of the log likelihood of a cell by a Poisson
    Notes: function assumes arg is the rate
            beta is set to zero (for now)
    """
    
    def __init__(self, loglike):
        """loglike: a LogLike object"""
        
        fmax=max(0, loglike.solve()[0])
        ## NB: the dd=-10 is a kluge for very small limits, set for loglike stuff with different scales.
        # this seems to work, but must be looked at more carefully
        self.pf = poisson.PoissonFitter(loglike, fmax=fmax, dd=-10.)
        self.loglike = loglike
        self.poiss=self.pf.poiss
        p = self
        self.fit= dict(t=loglike.t, tw=loglike.tw, fexp=loglike.fexp, 
                       flux=np.round(p.flux,4), 
                       errors=np.abs(np.array(p.errors)-p.flux).round(3),
                       limit=np.round(p.limit, 3), 
                       ts=np.round(p.ts,3), 
                       poiss=self.poiss, 
                      ) 

    def __call__(self, flux):
        return self.poiss(flux)
    
    def __repr__(self):
        relerr = np.abs(np.array(self.errors)/self.flux-1) if self.flux>0 else [0,0]
        return f'{self.__class__.__module__}.{self.__class__.__name__}: flux: {self.flux:.3f}[1+{relerr[0]:.2f}-{relerr[1]:.2f}], ' \
               f'limit: {self.limit:.2f}, ts: {self.ts:.1f}'
    @property
    def flux(self):
        return self.poiss.flux
    @property
    def errors(self):
        return self.poiss.errors
    @property
    def limit(self):
        """ 95% confidence interval"""
        return self.poiss.cdfcinv(0.05)
    @property
    def ts(self):
        return self.poiss.ts
    
    def comparison_plots(self, xlim=(0,1), ax=None, nbins=40):
        """Plots comparing this approximation to the actual likelihhod
        """
        f_like = lambda x: self(x)
        #pr = light_curve.PoissonRep(self);
        fp = lambda x: self(x)
        f_like = lambda x: self.loglike([x])
        fi = self.loglike.fit_info()
        xp = fi['flux']
        sigp = fi['sig_flux']
        peak = f_like(xp)
        dom = np.linspace(xlim[0],xlim[1],nbins)
        f_gauss = lambda x: -((x-xp)/sigp)**2/2
        fig, ax = plt.subplots(figsize=(6,4)) if not ax else (ax.figure, ax)
        ax.plot(dom, [f_like(x)-peak for x in dom], '-', label='Actual Likelihood');
        ax.plot(dom, fp(dom), '--+', lw=1, label='Poisson approximation');
        ax.plot(dom, [f_gauss(x) for x in dom], ':r', label='Gaussian approximation');
        ax.grid(alpha=0.5);
        ax.set(ylim=(-9,0.5));
        ax.axhline(0, color='grey', ls='--')
        ax.legend()
        

class PoissonRepTable(PoissonRep):
    
    def __init__(self, loglike):
        # PoissonRep fits to Poisson
        super().__init__(loglike, )
        # now make a table and add to dict
        dom,cod = self.create_table()
        self.fit['dom']=dom
        self.fit['cod']=cod.astype(np.float32)
        
    def create_table(self, npts=100, support=1e-6):
        # make a table of evently-spaced points between limits
        p = self.fit['poiss']
        a,b = p.cdfinv(support), p.cdfcinv(support)
        dom=(a,b,npts)
        cod = np.array(list(map(p, np.linspace(*dom)))) .astype(np.float32)
        return dom, cod
        
        
class LightCurve(object):
    """ In the language of Kerr, manage a set of cells
    """
    defaults=(
        ('min_exp', 0.3, 'mimimum exposure factor'),
        ('rep',   'poisson', 'name of the likelihood representation: poisson, gauss, or gauss2d'),
        ('replist', 'gauss gauss2d poisson poisson_table'.split(), 'Possible reps'),
        ('rep_class', [GaussianRep, Gaussian2dRep, PoissonRep, PoissonRepTable], 'coresponding classes'),
    )
    @keyword_options.decorate(defaults)
    def __init__(self, binned_weights, **kwargs):
        """Load binned data
        parameters:
            binned_weights : an iterable object that is a list of dicts; expect each
                to have following keys:
                    t, tw, fexp, w, S, B
            min_exp : minimum fractional exposure allowed
        """
        keyword_options.process(self,kwargs)
        global data
        data=binned_weights.data

        # select the set of cells 
        self.cells = [LogLike(ml) for ml in binned_weights if ml['fexp']>self.min_exp] 
        if data.verbose>0:
            print(f'Loaded {len(self)} / {len(binned_weights)} cells with exposure >'\
                  f' {self.min_exp} for light curve analysis')
        
        # analyze using selected rep
        if self.rep not in self.replist:
            raise Exception(f'Unrecognized rep: "{self.rep}", must be one of {self.replist}')
        repcl = self.rep_class[self.replist.index(self.rep)]
        try:
            self.fit_df = self.fit(repcl)
        except Exception as msg:
            print(f'Fail fit: {msg}')
         
    def __repr__(self):
        return f'{self.__class__} {len(self.cells)} cells fit with rep {self.rep}'
    
    def __getitem__(self, i):
        return self.cells[i]
    
    def __len__(self):
        return len(self.cells)
    
    def fit(self, repcl):
        """Perform fits to all intervals with chosen log likelihood representataon
        """
        outd = dict()
        for i, cell in enumerate(self):
            try:
                outd[i]=repcl(cell).fit
            except Exception as msg:
                print(f'{repcl} fail for Index {i}, LogLike {cell}\n   {msg}')
                raise

        df = pd.DataFrame.from_dict(outd, orient='index', dtype=np.float32)
        if data.verbose>0:
            print(f'Fits using representation {self.rep}: {len(self)} intervals\n  columns: {list(df.columns)} ')
        return df 

    def flux_plot(self, ts_max=9, xerr=0.5, title=None, ax=None, **kwargs): 
        """Make a plot of flux with according to the representation
        """
        kw=dict(yscale='linear',xlabel='MJD', ylabel='relative flux',)
        kw.update(**kwargs)
        df=self.fit_df
        if self.rep=='poisson':
            ts = df.ts
            limit = ts<ts_max
            bar = df.loc[~limit,:]
            lim = df.loc[limit,:]
        else: 
            bar=df; lim=[]
        
        fig, ax = plt.subplots(figsize=(12,4)) if ax is None else (ax.figure, ax)
        
        # the points with error bars
        t = bar.t
        xerr = bar.tw/2
        y =  bar.flux.values
        if self.rep=='poisson':
            dy = [bar.errors.apply(lambda x: x[i]).clip(0,4) for i in range(2)]
        elif self.rep=='gauss' or self.rep=='gauss2d':
            dy = bar.sig_flux.clip(0,4)
        else: assert False     
        ax.errorbar(x=t, y=y, xerr=xerr, yerr=dy, fmt='+')
        
        # now do the limits
        if len(lim)>0:
            t = lim.t
            xerr = lim.tw/2
            y = lim.limit.values
            yerr=0.2*(1 if kw['yscale']=='linear' else y)
            ax.errorbar(x=t, y=y, xerr=xerr,
                    yerr=yerr,  color='C1', 
                    uplims=True, ls='', lw=2, capsize=4, capthick=0,
                    alpha=0.5)
        
        #ax.axhline(1., color='grey')
        ax.set(**kw)
        ax.set_title(title or f'{data.source_name}, rep {self.rep}')
        ax.grid(alpha=0.5)
        
    def fit_hists(self, title=None, **hist_kw):
        """ Generate set of histograms of rate, error, pull, and maybe TS
        """
        hkw = dict(log=True, histtype='stepfilled',lw=2, edgecolor='blue', facecolor='lightblue')
        hkw.update(hist_kw)

        df = self.fit_df
        fig, (ax1,ax2,ax3)= plt.subplots(1,3, figsize=(12,2.5))
        x = df.t
        y = df.flux
        if self.rep=='poisson': #mean of high, low 
            yerr = df.errors.apply(lambda x: (x[0]+x[1])/2).clip(0,4) 
        elif self.rep=='gauss' or self.rep=='gauss2d':
            yerr = df.sig_flux.clip(0,4)
        else: assert False

        def shist(ax, x,  xlim, nbins, label, xlog=False): 
            def space(xlim, nbins=50):
                if xlog:
                    return np.logspace(np.log10(xlim[0]), np.log10(xlim[1]))
                return np.linspace(xlim[0], xlim[1], nbins+1)
            info = f'mean {x.mean():6.3f}\nstd  {x.std():6.3f}'
            ax.hist(x.clip(*xlim), bins=space(xlim, nbins), **hkw)
            ax.set(xlabel=label, xscale='log' if xlog else 'linear', ylim=(0.8,None))
            ax.text(0.65, 0.84, info, transform=ax.transAxes,fontdict=dict(size=10, family='monospace')) 
            ax.grid(alpha=0.5)
            return ax

        shist(ax1, y, (0.2, 5), 25, 'relative flux', xlog=True).axvline(1, color='grey')
        shist(ax2, yerr, (1e-2, 0.3), 25, 'sigma', xlog=True)
        shist(ax3, (y-1)/yerr, (-6,6), 25,'pull').axvline(0,color='grey')
        fig.suptitle(title or  f'{data.source_name}, rep {self.rep}')
        

       
class MyFitness(FitnessFunc):
    
    def __init__(self, p0=0.05, gamma=None, ncp_prior=None,):
        #super().__init__(p0, gamma, ncp_prior)
        self.ncp_prior = ncp_prior
        
    def fitness(self, R): # N_k, w_k):
        """ For cells [0,R) return array of length R+1 of the maximum likelihoods for combined cells 
        0..R, 1..R, ... R
        """
        # exposures and corresponding counts
        w_k = self.block_length[:R + 1] - self.block_length[R + 1]
        N_k = np.cumsum(self.x[:R + 1][::-1])[::-1]
        
        # eq. 26 from Scargle 2012
        return N_k * (np.log(N_k) - np.log(w_k))
    
    def fit(self, t, x, sigma):
        """Fit the Bayesian Blocks model given the specified fitness function.

        Parameters
        ----------
        t : array_like
            data times (one dimensional, length N)
            here the cumulative exposure, corresponding to Scargle 2012 Section 3.2
        x : array_like 
            data values, here the counts per bin
        Returns
        -------
        edges : ndarray
            array containing the (M+1) edges defining the M optimal bins
        """
        t, self.x, sigma = self.validate_input(t, x)

        # create length-(N + 1) array of cell edges
        edges = np.concatenate([t[:1],
                                0.5 * (t[1:] + t[:-1]),
                                t[-1:]])
        self.block_length = t[-1] - edges

        # arrays to store the best configuration
        N = len(t)
        best = np.zeros(N, dtype=float)
        last = np.zeros(N, dtype=int)

        # ----------------------------------------------------------------
        # Start with first data cell; add one cell at each iteration
        # ----------------------------------------------------------------
        for R in range(N):

            # evaluate fitness function
            fit_vec = self.fitness(R)

            A_R = fit_vec - self.ncp_prior
            A_R[1:] += best[:R]

            i_max = np.argmax(A_R)
            last[R] = i_max
            best[R] = A_R[i_max]

        # ----------------------------------------------------------------
        # Now find changepoints by iteratively peeling off the last block
        # ----------------------------------------------------------------
        change_points = np.zeros(N, dtype=int)
        i_cp = N
        ind = N
        while True:
            i_cp -= 1
            change_points[i_cp] = ind
            if ind == 0:
                break
            ind = last[ind - 1]
        change_points = change_points[i_cp:]

        return edges[change_points]
    
    
class BayesianBlocks(object):
    """
    """
    def __init__(self, data, min_exp=0.1):
        """
        data : a main.Main object with access to data for a given source
        """
        # Create 1-day cells from data
        self.data =data
        self.verbose=data.verbose
        
        self.bw = data.binned_weights()
        self.cells = list(filter(lambda x: x['fexp']>min_exp, self.bw))
        
        if self.verbose>0:
            print(f'Selected {len(self.cells)} / {len(self.bw)} with exposure > {min_exp}')
           
    def partition(self, p0=0.05, **kwargs):
        """
        Partition the interval into blocks using counts and cumulative exposure
        Return a BinnedWeights object using the partition
        """
        
        self.nn = np.array([len(cell['w']) for cell in self.cells]) #number per cell

        assert min(self.nn)>0, 'Attempt to Include a cell with no contents'

        mjd = np.array([cell['t'] for cell in self.cells])        
        self.cumexp = np.cumsum( [cell['fexp'] for cell in self.cells]  )
        N = len(mjd)
        ncp_prior = FitnessFunc(p0).compute_ncp_prior(N)
                  
        # Now run the astropy Bayesian Blocks code using my version of the 'event' model
        exp_edges = bayesian_blocks(t=self.cumexp, 
                                    x=self.nn.astype(float),
                                    fitness=MyFitness, ncp_prior=ncp_prior)
        if self.verbose>0:
            print(f'Partitioned {N} cells into {len(exp_edges)-1} blocks, with prior {ncp_prior:.1f}\n'\
                  f' Used FitnessFunc class {MyFitness} ' )
        
        # convert back from cumexp to mjd and rebin data
        mjd_index = np.searchsorted(self.cumexp, exp_edges)
        mjd_edges = mjd[mjd_index]
        return self.data.binned_weights(mjd_edges)
        
    def light_curve(self, bw=None, rep='poisson', min_exp=0.1):
        """ Return a LightCurve object using the specified BinnedWeights object, or default to the daily one.
        """        
        return LightCurve(bw or self.bw, rep=rep, min_exp=min_exp)