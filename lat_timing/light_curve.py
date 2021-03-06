"""Manage the binned time data to produce light curves, etc.

"""
import numpy as np
import pylab as plt
import pandas as pd
import os, sys, pickle
import matplotlib.ticker as ticker

from scipy import (optimize, linalg)
from scipy.linalg import (LinAlgError, LinAlgWarning)

from utilities import  keyword_options
from . import poisson

data=None # data_managment.Data obect for access to name, verbose, etc.

poisson_tolerance=0.2
    
class LogLike(object):
    """ implement Kerr Eqn 2 for a single interval, or cell"""
    
    def __init__(self, cell):
        """ cell is a dict"""       
        self.__dict__.update(cell)
        assert self.n>0, f'No data for cell {cell}'
        
    def fit_info(self, fix_beta=True):
        """Perform fits, return a dict with cell info"""
        pars = self.solve(fix_beta)
        if pars is None:
            if self.verbose>0:
                print(f'Fail fit for {self}')
            #return None
            raise RuntimeError(f'Fit failure: {self}')
        hess = self.hessian(pars)
        outdict = dict(t=self.t, tw=self.tw, fexp=self.fexp, counts=len(self.w) )
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
            pars: array or float
                if array with len>1, expect (rate, beta)
        """
        pars = np.atleast_1d(pars)
        if len(pars)>1:      alpha, beta = pars - np.array([-1,0])
        else:                alpha, beta = max(-1, pars[0]-1), 0
            
        tmp =  1 + alpha*self.w + beta*(1-self.w)  
        # limit alpha
        tmp[tmp<=1e-6]=1e-6
  
        return np.sum( np.log(tmp)) - alpha*self.S - beta*self.B

    def __repr__(self):
        return f'{self.__class__.__module__}.{self.__class__.__name__}:'\
        f' time {self.t:.3f}, {self.n} weights,  exposure {self.fexp:.2f}, S {self.S:.0f}, B {self.B:.0f}'
        
    def gradient(self, pars ):
        """gradient of the log likelihood with respect to alpha=flux-1 and beta, or just alpha
        """
        w,S = self.w, self.S
        pars = np.atleast_1d(pars)
  
        alpha =  max(-0.999,pars[0] -1)        
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
        alpha = max(-0.999, pars[0]-1)
        if  len(pars)==1:
            D = 1 + alpha*w 
            return [np.sum((w/D)**2)]
        else:
            beta= pars[1]
            Dsq = (1 + alpha*w + beta*(1-w))**2
            a, b, c = np.sum(w**2/Dsq), np.sum(w*(1-w)/Dsq), np.sum((1-w)**2/Dsq)
            return np.array([[a,b], [b,c]])
        
    def rate(self, fix_beta=True, debug=False, no_ts=False):
        """Return signal rate and its error"""
        try:
            s = self.solve(fix_beta )
            if s is None:
                return None
            h = self.hessian(s)
        
            v = 1./h[0] if fix_beta else linalg.inv(h)[0,0]
            ts = None if no_ts else (0 if s[0]<=-1 else
                 2*(self(s)-self( -1 if fix_beta else [-1,s[1]] )))
            return (s[0]), np.sqrt(v), ts
        
        except (LinAlgError, LinAlgWarning, RuntimeWarning) as msg:
            if debug or self.verbose>2:
                print(f'Fit error, cell {self},\n\t{msg}')
        except Exception as msg:
            print(f'exception: {msg}')
            raise
        print( '***********Failed?')
            
    def minimize(self,   fix_beta=True,estimate=[0.,0], **fmin_kw):
        """Minimize the -Log likelihood """
        kw = dict(disp=False)
        kw.update(**fmin_kw)
        f = lambda pars: -self(pars)
        return optimize.fmin_cg(f, estimate[0:1] if fix_beta else estimate, **kw)

    def solve(self, fix_beta=True, debug=True, estimate=[0.1,0],**fit_kw):
        """Solve non-linear equation(s) from setting gradient to zero 
        note that the hessian is a jacobian
        """

        if fix_beta:
            # 
            g0= self.gradient([0])
            # solution is at zero flux
            if g0<=0:
                return [0]
            # check that solution close to zero, difficult for fsolve.
            # if < 0.5 sigma away, just give linear solution  
            h0=self.hessian(0)[0]
            if g0/h0 < 0.5*np.sqrt(1/h0): 
                return [g0/h0]
       
        kw = dict(factor=2, xtol=1e-3, fprime=self.hessian)
        kw.update(**fit_kw)        
        try:
            ret = optimize.fsolve(self.gradient, estimate[0] if fix_beta else estimate , **kw)   
        except RuntimeWarning as msg:
            if debug or self.verbose>2:
                print(f'Runtime fsolve warning for cell {self}, \n\t {msg}')
            return None
        except Exception as msg:
            raise Exception(msg)
        return np.array(ret)
        
    def plot(self, fix_beta=True, xlim=(0,1.2),ax=None, title=None):
        fig, ax = plt.subplots(figsize=(4,2)) if ax is None else (ax.figure, ax)

        dom = np.linspace(*xlim)
        if fix_beta:
            f = lambda x: self([x])
            beta=0
        else:
            a, beta = self.solve(fix_beta, debug=True)
            f = lambda x: self([x, beta])
        ax.plot(dom, list(map(f,dom)) )
        
        try:
            a, s, ts = self.rate(fix_beta=fix_beta, debug=True)
            ax.plot(a, f(a), 'or')
            ax.plot([a-s, a+s], [f(a-s), f(a+s)], '-k',lw=2)
            for x in (a-s,a+s):
                ax.plot([x,x], [f(x)-0.1, f(x)+0.1], '-k',lw=2)
            ax.plot(a, f(a)-0.5, '-ok', ms=10)
            ax.set(title=title, xlim=xlim, ylim=(f(a)-4, f(a)+0.2), 
               ylabel='log likelihood', xlabel='flux')        
        except Exception as msg :
            print(msg)
            ax.set(title=' **failed fit**')
        ax.grid()


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
    
    def __init__(self, loglike, tol=poisson_tolerance # note global
                ):
        """loglike: a LogLike object"""
        
        rate, sig, ts= loglike.rate(no_ts=True)
#         if t is None:
#             raise Exception('Failed fit?')
        fmax=max(0, rate)
        ## NB: the dd=-10 is a kluge for very small limits, set for loglike stuff with different scales.
        # this seems to work, but must be looked at more carefully
        try:
            self.pf = poisson.PoissonFitter(loglike, fmax=fmax, scale=sig if rate>0 else 1,  dd=-10., tol=tol)
        except Exception as msg:
            print(f'Fail poisson fit for {loglike}: {msg}')
            with open('failed_loglike.pkl', 'wb') as file:
                pickle.dump(loglike, file)
            print('Saved file')
            raise
        self.loglike = loglike
        self.poiss=self.pf.poiss
        p = self
        self.fit= dict(t=loglike.t, tw=loglike.tw, counts=len(loglike.w), 
                       fexp=loglike.fexp, 
                       flux=np.round(p.flux,4), 
                       errors=np.abs(np.array(p.errors)-p.flux).round(3),
                       limit=np.round(p.limit, 3), 
                       ts=np.round(p.ts,3), 
                       poiss_pars=list(np.float32(self.poiss.p)), 
                      ) 

    def __call__(self, flux):
        return self.poiss(flux)
    
    def __repr__(self):
        relerr = np.abs(np.array(self.errors)/self.flux-1) if self.flux>0 else [0,0]
        return  f'{self.__class__.__module__}.{self.__class__.__name__}:'\
                f' flux: {self.flux:.3f}[1+{relerr[0]:.3f}-{relerr[1]:.3f}], '\
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
    
    def create_table(self, npts=100, support=1e-6):
        # make a table of evently-spaced points between limits
        pars = self.fit['poiss_pars']
        p = poisson.Poisson(pars)
        a,b = p.cdfinv(support), p.cdfcinv(support)
        dom=(a,b,npts)
        cod = np.array(list(map(p, np.linspace(*dom)))) .astype(np.float32)
        return dom, cod
    
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
        self.dom,self.cod = self.create_table()

    def __call__(self, x):
        return np.interp(x, np.linspace(*self.dom), self.cod)

        
class LightCurve(object):
    """ In the language of Kerr, manage a set of cells
    """
    defaults=(
        ('min_exp', 0.3, 'mimimum exposure factor'),
        ('rep',   'poisson', 'name of the likelihood representation: poisson, gauss, or gauss2d'),
        ('replist', 'gauss gauss2d poisson poisson_table'.split(), 'Possible reps'),
        ('rep_class', [GaussianRep, Gaussian2dRep, PoissonRep, PoissonRepTable], 'coresponding classes'),
        ('verbose', 0, 'verbosity'),
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
        self.data=data=binned_weights.data
        self.source_name = self.data.source_name
        self.verbose = getattr(data, 'verbose', self.verbose)

        # select the set of cells 
        self.cells = [LogLike(ml) for ml in binned_weights if ml['fexp']>self.min_exp] 
        if self.verbose>0:
            print(f'Loaded {len(self)} / {len(binned_weights)} cells with exposure >'\
                  f' {self.min_exp} for light curve analysis')
        
        # analyze using selected rep
        if self.rep not in self.replist:
            raise Exception(f'Unrecognized rep: "{self.rep}", must be one of {self.replist}')
        repcl = self.rep_class[self.replist.index(self.rep)]
        
        self.fit_df = self.fit(repcl)
#         try:
#             self.fit_df = self.fit(repcl)
#         except Exception as msg:
#             print(f'Fail fit: {msg} ') #'; stop here')
#             raise Exception(msg)
         
    def __repr__(self):
        return f'{self.__class__.__name__}: source "{self.source_name}" fit with {len(self.fit_df)} cells'
    
    def to_dict(self):
        return dict(source_name=self.source_name, 
                    rep=self.rep,
                    edges = self.data.edges,
                    fit_dict = self.fit_df.to_dict('records'),
               )


    def write(self, filename):
        with open(filename, 'wb') as out:
            pickle.dump( self.to_dict(), out)

            if self.verbose>0:
                print(f'Wrote light curve for source "{self.source_name}" to {filename}')

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
                print(f'{self.__class__}, {repcl} fail for Index {i}, LogLike {cell}\n   {msg}')
                raise

        df = pd.DataFrame.from_dict(outd, orient='index', dtype=np.float32)
        if self.verbose>0:
            print(f'Fits using representation {self.rep}: {len(self)} intervals')
            if self.verbose>1:
                print(f'    columns: {list(df.columns)} ')
        return df 

    @property
    def dataframe(self):
        """return the summary DataFrame
        """
        return self.fit_df
    
    def create_tables(self, npts=100, support=1e-6):
        """ create a set of tables representing the log-likelihoods of the cells
        parameters:
            npts : integer
                number of points
            support : float
                Integral of the Likelihood PDF to discard on each end
            
        returns 
            arrays for points and log-likelihood values as (N, npts) arraays
        """
        assert self.rep.startswith('poisson'), 'Need a Poisson rep to calculate tables'
        df = self.dataframe
        dom = np.empty((len(df), npts), np.float32)
        cod = np.empty((len(df), npts), np.float32)

        def create_table(poiss):
            # make a table of evently-spaced points between limits
            a = poiss.cdfinv(support) if poiss.flux>0 else 0
            b = poiss.cdfcinv(support)
            x = np.linspace(a,b,npts).astype(np.float32)
            y = np.array(list(map(poiss, x))) .astype(np.float32)
            return x, y

        for i,poiss in enumerate(map( poisson.Poisson, df.poiss_pars.values)):
            dom[i], cod[i] = create_table(poiss)
        return dom, cod
    
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
        
        fig, ax = plt.subplots(figsize=(12,4)) if ax is None else (ax.figure, ax)\
            if ax is not None else (ax.figure,ax)
        
        # the points with error bars
        t = bar.t
        xerr = bar.tw/2
        y =  bar.flux.values
        if self.rep=='poisson':
            dy = [bar.errors.apply(lambda x: x[i]).clip(0,4) for i in range(2)]
        elif self.rep=='gauss' or self.rep=='gauss2d':
            dy = bar.sig_flux.clip(0,4)
        else: assert False     
        ax.errorbar(x=t, y=y, xerr=xerr, yerr=dy, fmt=' ', color='silver')
        
        # now do the limits
        if len(lim)>0:
            t = lim.t
            xerr = lim.tw/2
            y = lim.limit.values
            yerr=0.2*(1 if kw['yscale']=='linear' else y)
            ax.errorbar(x=t, y=y, xerr=xerr,
                    yerr=yerr,  color='lightsalmon', 
                    uplims=True, ls='', lw=2, capsize=4, capthick=0,
                    alpha=0.5)
        
        #ax.axhline(1., color='grey')
        ax.set(**kw)
        ax.set_title(title or f'{self.source_name}, rep {self.rep}')
        ax.grid(alpha=0.5)
        
    def fit_hists(self, title=None, fignum=None, **hist_kw):
        """### Plot distributions of rate, error, pull
        
        Source name:  "{source_name}" 
        Bins fit with {self.rep} model.
        
        {fig.html}
        
        """
        source_name=self.source_name
        hkw = dict(log=True, histtype='stepfilled',lw=2, edgecolor='blue', facecolor='lightblue')
        hkw.update(hist_kw)

        df = self.fit_df
        fig, (ax1,ax2,ax3)= plt.subplots(1,3, figsize=(10,3.0), tight_layout=True, num=fignum)
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
            ax.text(0.62, 0.82, info, transform=ax.transAxes,fontdict=dict(size=10, family='monospace')) 
            ax.grid(alpha=0.5)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda val,pos: { 1.0:'1', 10.0:'10', 100.:'100', 5:'5', -5:'-5'}.get(val,'')))
            return ax

        shist(ax1, y, (0.25, 4), 25, 'relative flux', xlog=True).axvline(1, color='grey')
        shist(ax2, yerr*100, (1, 30), 25, 'sigma [%]', xlog=True)
        shist(ax3, (y-1)/yerr, (-6,6), 25,'pull').axvline(0,color='grey')
        return fig

    def mean_std(self):
        """ return weighted mean and rms"""
        df = self.fit_df
        frms = df.errors.apply(lambda err: 0.5*(err[0]+err[1]))
        fwts = 1/frms**2
        fmean = np.sum(df.flux*fwts)/np.sum(fwts)
        t = (df.flux-fmean)/frms
        return fmean, t.std()

class LightCurveX(LightCurve):
    """ subclass of LightCurve instantiated from file or dict generated by it
    """
    def __init__(self, input:'filename or dict', verbose=1):

        self.verbose = verbose
        
        def load_from_dict(t):
            self.source_name = t.get('source_name', '')
            self.rep = t['rep']
            self.fit_df = pd.DataFrame(t['fit_dict'])

        if type(input)==dict:
            load_from_dict(input)
        else:
            with open(input, 'rb') as inp:
                load_from_dict(pickle.load(inp))
            