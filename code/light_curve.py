"""Manage the binned time data to produce light curves, etc.

"""
import numpy as np
import pylab as plt
import pandas as pd
import os, sys

from scipy import (optimize, linalg)
from scipy.linalg import (LinAlgError, LinAlgWarning)
import keyword_options, poisson

data=None # data_managment.Data obect for access to name, verbose, etc.



class LightCurve(object):
    """ In the language of Kerr, manage a set of cells
    """
    defaults=(
        ('min_exp', 0.3, 'mimimum exposure factor'),
        ('rep',   'poiss', 'name of the likelihood representation: poiss, gauss, or gauss2d'),

    )
    @keyword_options.decorate(defaults)
    def __init__(self, binned_weights, **kwargs):
        """Load binned data
        parameters:
            binned_weights : an iterable object that is a list of dicts; expect each to have
                 keys t, exp, w, S, B
            min_exp : filter by exposure factor
        """
        keyword_options.process(self,kwargs)
        global data
        data=binned_weights.data
        self.cells = [LogLike(ml) for ml in binned_weights if ml['exp']>self.min_exp] 
        if data.verbose>0:
            print(f'Loaded {len(self)} / {len(binned_weights)} cells with exposure > {self.min_exp} for light curve analysis')
        self.representation = None
         
    def __repr__(self):
        return f'{self.__class__} {len(self.cells)} cells loaded'
    
    def __getitem__(self, i):
        return self.cells[i]
    
    def __len__(self):
        return len(self.cells)
    
    def fit(self, **kwargs):
        """Perform fits to all intervals assuming likelihood are normally distributed,
                set a DataFrame with results
        """
        outd = dict()
        if self.rep=='gauss':
            
            r=[]; bad=[]; good=[]; ts=[]
            for i,cell in enumerate(self):

                outd[i] = cell.info(fix_beta=True)

            self.fit_df=  pd.DataFrame.from_dict(outd, orient='index', dtype=np.float32)

        elif self.rep=='poiss':
            self.poiss_fit(**kwargs)
        else:
            assert False

    def poiss_fit(self, **kwargs):
        """ fit using poisson fitter  
        
        """
        pdict=dict()
        for i,q in enumerate(self):
            try:
                p = PoissonRep(q) #.fit_poisson().poiss

                pdict[i] = dict(t=q.t, flux=np.round(p.flux,4), exp=q.exp, 
                                errors=np.abs(np.array(p.errors)-p.flux).round(3),
                                limit=np.round(p.limit, 3), ts=np.round(p.ts,3), funct=p ) 
            except Exception as msg:
                print(f'Fail for Index {i}, LogLike {q}\n   {msg}')
                raise
        self.fit_df = pd.DataFrame.from_dict(pdict,orient='index', dtype=np.float32)  
        if data.verbose>0:
            print(f'Fit {len(self)} intervals: columns (t, exp, flux, errors, limit, ts, funct) in a DataFrame.')
                
    def flux_plot(self, title=None, ax=None): 

        df=self.fit_df
        t = df.t
        y=  df.flux.values.clip(0,4)
        if self.rep=='poiss':
            dy = [df.errors.apply(lambda x: x[i]).clip(0,4) for i in range(2)]
        elif self.rep=='gauss':
            dy = df.sig_flux.clip(0,4)
        else:
            raise Exception(f'unrecognized fitter: {rep}')

        if not ax:
            fig, ax = plt.subplots(figsize=(12,4))
        else:
            fig =ax.figure
        ax.errorbar(x=t, y=y, yerr=dy, fmt='+')
        ax.axhline(1., color='grey')
        ax.set(xlabel='MJD', ylabel='relative rate')
        ax.set_title(title or data.source_name)
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
        yerr = df.sig_flux

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

        shist(ax1, y, (0.2, 5), 25, 'relative rate', xlog=True).axvline(1, color='grey')
        shist(ax2, yerr, (1e-2, 0.3), 25, 'sigma', xlog=True)
        shist(ax3, (y-1)/yerr, (-6,6), 25,'pull').axvline(0,color='grey')
        fig.suptitle(title or data.source_name+' fit summary')
    
class LogLike(object):
    """ implement Kerr Eqn 2 for a single interval, or cell"""
    
    def __init__(self, cell):
        """ cell is a dict"""
        
        self.__dict__.update(cell)
        self.estimate= [0.5, 0]
        
    def info(self, fix_beta=True):
        """Perform fits, return a dict with cell info"""
        pars = self.solve(fix_beta)
        if pars is None:
            if data.verbose>0:
                print(f'Fail fit for {self}')
            return None
            #raise RuntimeError('Fit failure')
        hess = self.hessian(pars)
        outdict = dict(t=self.t, exp=self.exp, counts=len(self.w) )
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
        return f'''{self.__class__}
        time {self.t:.3f}, {len(self.w)} weights,  exposure {self.exp:.2f}, S {self.S:.0f}, B {self.B:.0f}'''
        

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
            
    def minimize(self,   fix_beta=True, **fmin_kw):
        """Minimize the -Log likelihood """
        kw = dict(disp=False)
        kw.update(**fmin_kw)
        f = lambda pars: -self(pars)
        return optimize.fmin_cg(f, self.estimate[0:1] if fix_beta else self.estimate, **kw)

    def solve(self, fix_beta=True, debug=False, estimate=[0.5,1],**fit_kw):
        """Solve non-linear equation(s) from setting gradient to zero 
        note that the hessian is a jacobian
        """
        kw = dict(factor=2, xtol=1e-3, fprime=self.hessian)
        kw.update(**fit_kw)

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


class GaussionRep(object):
    pass #TODO put stuff that assumes the simple gaussian representation
        
class PoissonRep(object):
    """Manage the representation of the log likelihood of a cell by a Poisson
    Notes: function assumes arg is the rate
            beta is set to zero
    """
    
    def __init__(self, loglike):
        """loglike: a LogLike object"""
        
        self.ll=loglike
        fmax=max(0, loglike.solve()[0])
        self.pf = poisson.PoissonFitter(loglike, fmax=fmax)
        self.poiss=self.pf.poiss

    def __call__(self, flux):
        return self.poiss(flux)
    
    def __repr__(self):
        t = np.array(self.errors)/self.flux-1
        relerr = np.abs(np.array(self.errors)/self.flux-1)
        return f'{self.__class__} flux: {self.flux:.3f}[1+{relerr[0]:.2f}-{relerr[1]:.2f}], ' \
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