"""
package dev initialization
"""
import os, sys
import pickle
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

from jupydoc import Publisher, DocPublisher
from lat_timing import (TimedDataX, LightCurve, LightCurveX, GetWeightedData)
# from lat_timing import (Main, LightCurve, WeightedDataX, BinnedWeights)

__docs__ = ['GammaData', 'DataDescription']

   
class GammaData(DocPublisher): 
    """
    title: Photon data setup
   
    sections:
         data_summary [read_data get_weights photon_data cell_data light_curve_data ] 

    default_source_name: ''
    data_path: $HOME/work/lat-data/sources

    # Remote information
    server:     rhel6-64.slac.stanford.edu
    username:   burnett
    remote_data_path: /nfs/farm/g/glast/u/burnett/work/lat/data/photons
    generation_path: /nfs/farm/g/glast/g/catalog/pointlike/skymodels


    """
            
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # source_name is the version
        self.source_name = self.version or self.default_source_name
        self.doc_info['title'] += f'\nSource {self.version}'

        self.home = os.environ.get('HOME', '')
        self.data_path = os.path.expandvars(self.data_path)

    def data_summary(self):
        """Data reduction

        The data processing, from photon data to light curves, is described <a href={link}>here</a>.   
        
        #### Specified source name: **{self.source_name}**

        {text}
        Contents of the folder with source data `{self.source_data_path}`
      
        {contents}
        
        """

        #-------------------------------------------------------------------
        # check for data
        ok, text = self.check_data()
        if not ok: return text
        
        otherdoc, link = self.docman.client('DataDescription')
        contents = self.shell(f'ls -l {self.source_data_path}',)
        self.publishme()
        

    def check_data(self):    
        """
        
        """
        import pysftp as sftp
        host = os.environ.get('HOST', 'local')
        if not self.source_name:
            text='No source name specified: locally available:\n'
            text+= self.shell(f'ls {self.data_path}', monospace=False)
            return False, text

        text=f'Source "{self.source_name}" '
        self.source_data_path = f'{os.path.expandvars(self.data_path)}/{self.source_name}'
        if not os.path.isdir(self.source_data_path):
            # not in the local. Check remote
            text+='not found locally -- checking remote...'
            
            try:
                with sftp.Connection(self.server, self.username) as srv:
                    srv.chdir(self.remote_data_path)
                    flist =srv.listdir()
                    if self.source_name in flist:
                        text += f'\nFound on remote, copying...:'
                        dest = os.path.join(self.data_path, self.source_name)
        #                text += f' to {dest}'
                        if not os.path.isdir(dest):
                            os.makedirs(dest, exist_ok=True)
                        srv.chdir(self.source_name)
                        tocopy = srv.listdir()
                        for file in tocopy:
                            text += f'{file}, '
                            srv.get(file, dest+'/'+file)
                    else:
                        text += 'Not found on remote.'
                return text
            except Exception as e:
                print(text+ f'\n*** Failed attempt to get remote files: {e}', file=sys.stderr)
                return False, text
                

            if not os.path.isdir(self.generation_path):
                text += f'\n*** No photon data for source "{self.source_name}" -- cannot generate on this machine.\n'
                text += f'Sources available here:\n'
                text += self.shell(f'ls {self.data_path}', monospace=False)
                #print(text, file = sys.stderr)
                return False, text
        else:
            text += f'data  ok.'
        return True, text

    def read_data(self):
        filename = self.source_data_path+'/time_data.pkl'
  
        if os.path.isfile(filename):
            self.timed_data = TimedDataX(filename)
            return self.already_generated(filename)
        else:

            return self.generate(filename)

    def get_weights(self):
        """Weights

        This subsection summarizes data retrieved from tehe server {wd.server} for the pointlie skymodel {wd.skymodel}.

        {text}

        This SED shows the spectral model used to determine the weights.

        {sed_image}

        And here is its fit summary.

        {src_info}
        """
        with self.capture_print() as text:
            self.wd = wd = GetWeightedData(self.source_name)

        sed_image = self.image(wd.sed_image_file, width=150, caption=f'SED for {self.source_name}')    
        src_info = self.monospace(wd.src_info, 'source fit information')

        
        self.publishme()
        
    def already_generated(self, filename):
        """Read in cone-selected data

        `TimeData` serialization found as dict in file <samp>{filename}</samp>.

        """
        #-------------------------------------------------------------------
        self.publishme()

    def generate(self, filename):
        """Generate cone-selected data set

        Run `Main` to make cone selection from full database:
        Writing `TimeData` serialization, to file <samp>{filename}</samp>.

        Output: 
        {text}
        
        Writing `TimeData` serialization, to file <samp>{filename}</samp>.
        {contents}
        
        """
        from lat_timing import Main
        with self.capture_print() as text:
            tdata = Main( 
                 name=self.source_name, 
                 weight_file=\
                 '/nfs/farm/g/glast/g/catalog/pointlike/skymodels/P8_10years/uw9011/weight_files',
                 )

        ok = tdata is not None
        if ok:
            self.timed_data = tdata.timedata
            tdata.timedata.write(filename)
            contents = self.shell(f'ls -l {self.source_data_path}')

        self.publishme()
        return None if ok else 'Failure to reed data'

    def photon_data(self):
        """Photon data

        This is a list of every photon in a valid time range, with time in MJD, energy/event type indicated
        by the band, position on the sky by a pixel index, and the radius and weight as shown.
        {photons}

        {fig1}
        """
        photons = self.timed_data.photon_data
        fig1, axx = plt.subplots(1,3, num=1, figsize=(10,3))
        hkw = dict(bins=32, histtype='step', lw=2)
        
        def band(ax):
            t = hkw.copy()
            t.update(bins=np.linspace(0,32,33), log=True)
            ax.hist(photons.band,  **t)
        def weight(ax):
            ax.hist(photons.weight, **hkw)
        def radius(ax):
            ax.hist(photons.radius,  **hkw)
        for f, label, ax in zip(
                [band, weight, radius], 
                'band weight radius'.split(),
                axx.flatten()): 
            ax.set_xlabel(label, fontsize=12)
            f(ax)
        fig1.caption=f'{self.source_name} data features'
        #-------------------------------------------------------------------
        self.publishme()

    def cell_data(self):
        """Cell Data

        Uses `TimedData.binned_weights()` to return a `BinnedWeights` object
        {bw_info}
        The `BinnedWeights` contains the cell data for default time bins:

        {cells}
        """
        #--------------------------------------
        td = self.timed_data
        self.bw = td.binned_weights()
        bw_info = self.monospace(self.bw)

        self.cells = cells =  self.bw.dataframe
        #---------------------------------------
        self.publishme()
    
    def light_curve_data(self):
        """The 1-day Light Curve 

        A "light curve" is derived  from the cell data, after fitting each cell to the a poisson-like function.
        This is the contents of the `LightCurve` object with the poisson fitter. (This section caches the
        result of the fit in the file <samp>{filename}</samp>.)
        
        {light_curve_info}
        {lc_df}

        the columns are: 
        * $t, tw$: time and width in JD units; 
        * _counts_: number of photons
        * _fext_: exposure factor, normaized to the average for a day
        * _flux_: fit value
        * _errors_: (lower, upper)
        * _limit_: 95% limit
        * _ts_: the TS value
        * poiss_pars: the three poisson-like parameters used to detemine flux, errors, limit, ts

        The `LightCurve` object has a display of the fits:

        {fig1}
        And a summary:
        {fig2}

        """
        #-------------------------------------------------------------------
        filename = self.source_data_path + '/light_curve.pkl'
        if os.path.isfile(filename):
            self.light_curve = LightCurveX(filename)
        else:
            self.light_curve = LightCurve(self.bw)
            self.light_curve.write(filename)

        light_curve_info = self.monospace(self.light_curve)
        lc_df = self.light_curve.dataframe
        fig1, ax = plt.subplots(figsize=(12,4), num=1)
        self.light_curve.flux_plot(ax =ax)
        fig2 = self.light_curve.fit_hists(fignum=2)

        #-------------------------------------------------------------------
        self.publishme()

    def get_light_curve(self, bins=None, **kwargs):
        """ Return a LightCurve object, containing a table of fluxes and other cell info
        bins: a list of bin edges | integer | None
            if None use default
        (This a a copy from Main.light_curve for convenience)
        """
        bw = self.timed_data.binned_weights(bins)
        if bins is None and len(kwargs)==0: # set or use cached version unless new bins or a kwarg
            if not hasattr(self, 'basic_lc'):
                self.basic_lc = LightCurve(bw, **kwargs)
            return self.basic_lc
        return LightCurve(bw, **kwargs)

    def data_save(self):
        """Save the Data?

        {local_text}

        """
        #-------------------------------------------------------------------

        if not self.fileok: 

            lc = self.light_curve
            outdict = dict(
                    source_name=self.source_name, 
                    photons=self.gdata.photons.to_dict('records'),
                    light_curve =
                        dict(
                            rep=lc.rep,
                            edges = lc.data.edges,
                            fit_dict = lc.fit_df.to_dict('records'),
                            ),
                        )        
            with open(outfile, 'wb') as out:
                pickle.dump( outdict,  out)
            with open(outfile, 'rb') as inp:
                pkl = pickle.load(inp)
            pkl_keys = list(pkl.keys())
            local_text = f"""
                Saving the data, which was generated from the basic photon data set
                ```
                    lc = self.light_curve
                    outdict = dict(
                        source_name=self.source_name, 
                        photons=self.gdata.photons.to_dict('records'),
                        light_curve =
                            dict(
                                rep=lc.rep,
                                edges = lc.data.edges,
                                fit_dict = lc.fit_df.to_dict('records'),
                                ),
                            )   
                    ```
                    {pkl_keys}
                """
            
        else:
            local_text= f"""
            Not saving since read in from file
             """
  
        #-------------------------------------------------------------------           
        self.publishme()

class DataDescription(DocPublisher): 
    """
    #title: Photon Data Description
   
    """            
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def title_page(self): 
        r"""<h2> Photon Data Description </h2>
    
        Data selection, from raw data to a light curve, results in five representations:

        1. **Photon data**: Public photon data generated by the "L1" processing. This reconstruction of the actual 
        detector information provides the energy, direction, and time for each photon, recorded in the "FT1" files.

        2. **Energy and angular binning**: The binning used here is performed by `pointlike`. 
        Bands are defined acording to energy, 4/decade and event type, front or back. 
        Thus there are 32 such bands from 100 MeV to 1 TeV.
        All 100M photons were extracted from the FT1 files and put into a database, with each
        photon characterized by its time and integers specifying the band and position.
        <br><br>These data are currently at `/nfs/farm/g/glast/u/burnett/work/lat-data/photon_dataset/`,
        a folder containing currently 132 monthly files in  [parquet format](https://en.wikipedia.org/wiki/Apache_Parquet), about 21 GB each. 
        The code uses [`pyarrow`](https://arrow.apache.org/docs/python/index.html) for processing.

        3. **Cone selection**: For the study of a single source, a cone about its position to a radius 
        of 5 or 7 degrees, is used to select the data set to use here. A *weight* is assigned to each photon.
        This uses the `pointlike` model depending on the PSF, optimized with respect to the positions and 
        spectral  characterists of all sources. The weight is the probabality that an individual photon was 
        from the source. A basic assumption for studies of the time behavior of a source is that the background
        is **not** varying; this can be verified by fitting **both** signal and background components for
        each cell, see next.

        3. **cells**: This is the basic partition in time, for studying time behavior. We usually
        use days for this, the MJD unit. A final input is from the effective area and exposure, 
        to normaize counts to
        flux. Each cell has then: 
          * central time and time inteval, in MJD units
          * exposure factor $e$.
          * a set of weights $w$. 
        <br><br>
        5. **Light curve**: Each cell is used to make a measurement of the signal flux, and perhaps also the
        background, by optimizing an estimator function. Here we use the likelihood derived by
        M. Kerr. Per cell, it is:
        <br><br>
        $ \displaystyle\log\mathcal{L}(\alpha,\beta\ |\ w,e) = \sum_{w}  \log \big( 1 + \alpha\ w + \beta\ (1-w) \big) - e\ (\alpha\ S + \beta\ B) $
        <br>               
        Here the likelihood is a function of the relative flux parameters $\alpha$ and $\beta$, given the set
         of weights $w$ and fractional exposure $e$. Normalization is such that the averages of $\alpha$ and $\beta$ are 1 and 0.
        $S$ and $B$ are the sums $\sum{w}$ and $\sum{(1-w)}$ for the full dataset. For fiting, we normally fix $\beta$ to its expected zero, 
        and approximate the likelihood function to a Poisson-like function. All the 
        likelihood parameters are then derived from this.
        """
        #-------------------------------------------------------------------
        # check for data
        self.publishme()
