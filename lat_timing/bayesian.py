import numpy as np
from astropy.stats.bayesian_blocks import FitnessFunc
from .light_curve import LightCurve
from utilities import keyword_options


class CountFitness(FitnessFunc):
    """
    Adapted version of a astropy.stats.bayesian_blocks.FitnessFunc
    Considerably modified to give the `fitness function` access to the cell data.
    Currently just implements the Event model using exposure instead of time.
    
    """
    
    def __init__(self, lc, p0=0.05,):
        """lc  : a LightCurbe object, includeing DataFrame, including exposure (fexp) and counts (counts), 
            as well as a representation of the likelihood for each cell
        """
        self.p0=p0
        self.lc = lc
        self.df=df=lc.dataframe
        N = self.N = len(df)
        # Invoke empirical function from Scargle 2012 
        self.ncp_prior = self.p0_prior(N)
        
        #actual times for bin edges
        dt = df.tw[0]/2 # assum all the same
        self.mjd = np.concatenate([df.t.values-dt, [df.t.values[-1]+dt]] ) # put one at the end 
        self.setup()
        
    def setup(self):
        df = self.df
        # counts per cell
        self.nn = df.counts.values 
        assert min(self.nn)>0, 'Attempt to Include a cell with no contents'

        
        # edges and block_length use exposure as "time"
        fexp = df.fexp.values
        self.edges = np.concatenate([[0], np.cumsum(fexp)])

        # replaced this 
        #         self.edges = np.concatenate([t[:1],
        #                         0.5 * (t[1:] + t[:-1]),
        #                         t[-1:]])
        
        self.block_length = self.edges[-1] - self.edges
        
    def __call__(self, R): 
        """ The fitness function needed for BB algorithm 
        For cells 0..R return array of length R+1 of the maximum likelihoods for combined cells 
        0..R, 1..R, ... R
        """
        # exposures and corresponding counts
        w_k = self.block_length[:R + 1] - self.block_length[R + 1]
        N_k = np.cumsum(self.nn[:R + 1][::-1])[::-1]
        
        # eq. 26 from Scargle 2012
        return N_k * (np.log(N_k) - np.log(w_k))
    
    def fit(self):
        """Fit the Bayesian Blocks model given the specified fitness function.
        Refactored version using code from bayesian_blocks.FitnesFunc.fit
        Returns
        -------
        edges : ndarray
            array containing the (M+1) edges, in MJD units, defining the M optimal bins
        """
        # This is the basic Scargle algoritm, copied almost verbatum
        # ---------------------------------------------------------------
        
        # arrays to store the best configuration
        N = self.N 
        best = np.zeros(N, dtype=float)
        last = np.zeros(N, dtype=int)

        # ----------------------------------------------------------------
        # Start with first data cell; add one cell at each iteration
        # ----------------------------------------------------------------
        for R in range(N):

            # evaluate fitness function
            fit_vec = self(R)

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

        return self.mjd[change_points]

    
class LikelihoodFitness(CountFitness):
    """ Fitness function that uses the full likelihood
    """
    
    def __init__(self, lc, p0=0.05,):
        super().__init__(lc, p0)
        
    def setup(self):
        df = self.df
        N = self.N
        if 'dom' in df.columns:
            cnpt = df.dom[0][-1]
            self.cdom = np.empty((N, cnpt))
            self.ccod = np.empty((N, cnpt))
            for i in range(N):
                self.cdom[i]=np.linspace(*df.dom[i])
                self.ccod[i]=df.cod[i]
        else:
            self.cdom, self.ccod = self.lc.create_tables(npts=200, support=2e-9)

    def __call__(self, R, npt=100):

        x = np.linspace(self.cdom[R][0], self.cdom[R][-1], npt)
        y = np.zeros(npt)
        rv = np.empty(R+1)
        for i in range(R, -1, -1): 
            y += np.interp(x, self.cdom[i], self.ccod[i], left=-np.inf, right=-np.inf)
            amax = np.argmax(y)
            rv[i] =y[amax]
        return rv    

    
class BayesianBlocks(object):
    """Perform Bayesian Block analysis of the cells found in a light curve
    """
    defaults=(
        ('verbose', 1, 'verbosity'),
        ('fitness_func', 'counts', 'Type of fitness function to use'),
        ('func_names', 'counts likelihood'.split(), 'allowed functions'),
        ('func_classes', [CountFitness, LikelihoodFitness], 'implemented classes'),
        ('p0',      0.05, 'probability used to calcualate prior'),
    )
    
    @keyword_options.decorate(defaults)
    def __init__(self, lc, fitness_func=None, **kwargs):
        """
        lc : a  LIghtCurve object with a DataFrame, whcih which must have "poiss" column
        """
        keyword_options.process(self,kwargs)
        self.lc = lc
        self.data = lc.data
        self.cells = lc.dataframe
        assert 'poiss_pars' in self.cells.columns, 'Expect the dataframe ho have the Poisson representation'
        self.fitness_func = dict(zip(self.func_names, self.func_classes)).get(fitness_func, self.func_classes[0])
        if self.fitness_func is None:
            raise Exception(f'Valid names for fitness_func are: {self.func_names}')
          
    def partition(self, p0=0.05, **kwargs):
        """
        Partition the interval into blocks using counts and cumulative exposure
        Return a BinnedWeights object using the partition
        """
                 
        # Now run the astropy Bayesian Blocks code using my version of the 'event' model
        fitness = self.fitness_func(self.lc, p0=self.p0)
        edges = fitness.fit() 
        
        if self.verbose>0:
            print(f'Partitioned {fitness.N} cells into {len(edges)-1} blocks, with prior {fitness.ncp_prior:.1f}\n'\
                  f' Used FitnessFunc class {self.fitness_func} ' )
        
        return self.data.binned_weights(edges)
        
    def light_curve(self, bw=None, rep='poisson', min_exp=0.1):
        """ Return a LightCurve object using the specified BinnedWeights object,
        """        
        return LightCurve(bw, rep=rep, min_exp=min_exp)

class BayesianBlocks(object):
    """Perform Bayesian Block analysis of the cells found in a light curve
    """
    defaults=(
        ('verbose', 1, 'verbosity'),
        ('fitness_func', 'counts', 'Type of fitness function to use'),
        ('func_names', 'counts likelihood'.split(), 'allowed functions'),
        ('func_classes', [CountFitness, LikelihoodFitness], 'implemented classes'),
        ('p0',      0.05, 'probability used to calcualate prior'),
    )
    
    @keyword_options.decorate(defaults)
    def __init__(self, lc, fitness_func=None, **kwargs):
        """
        lc : a  LIghtCurve object with a DataFrame, whcih which must have "poiss" column
        """
        keyword_options.process(self,kwargs)
        self.lc = lc
        self.data = lc.data
        self.cells = lc.dataframe
        assert 'poiss_pars' in self.cells.columns, 'Expect the dataframe ho have the Poisson representation'
        self.fitness_func = dict(zip(self.func_names, self.func_classes)).get(fitness_func, self.func_classes[0])
        if self.fitness_func is None:
            raise Exception(f'Valid names for fitness_func are: {self.func_names}')
          
    def partition(self, p0=0.05, **kwargs):
        """
        Partition the interval into blocks using counts and cumulative exposure
        Return a BinnedWeights object using the partition
        """
                 
        # Now run the astropy Bayesian Blocks code using my version of the 'event' model
        fitness = self.fitness_func(self.lc, p0=self.p0)
        edges = fitness.fit() 
        
        if self.verbose>0:
            print(f'Partitioned {fitness.N} cells into {len(edges)-1} blocks, with prior {fitness.ncp_prior:.1f}\n'\
                  f' Used FitnessFunc class {self.fitness_func} ' )
        
        return self.data.binned_weights(edges)
        
    def light_curve(self, bw=None, rep='poisson', min_exp=0.1):
        """ Return a LightCurve object using the specified BinnedWeights object,
        """        
        return LightCurve(bw, rep=rep, min_exp=min_exp)

