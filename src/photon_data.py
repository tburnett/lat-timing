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

__docs__ = ['GammaData', 'DataDescription', 'DataFiles']

   
class GammaData(DocPublisher): 
    """
    title: Photon data setup
   
    sections:
         data_summary [read_data get_weights photon_data cell_data light_curve_data ] 

    default_source_name: ''
    data_path: $HOME/work/lat-data/sources

    # Remote (SLAC) information
    server:     rhel6-64.slac.stanford.edu
    username:   burnett
    remote_data_path: /nfs/farm/g/glast/u/burnett/work/lat-data/data/sources
    weight_path: /nfs/farm/g/glast/g/catalog/pointlike/skymodels/P8_10years/uw9011/weight_files


    """
            
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # source_name is the version
        self.source_name = self.version or self.default_source_name
        self.doc_info['title'] += f'\nSource {self.version}'

        self.home = os.environ.get('HOME', '')
        self.wsl = self.home.startswith('/home')
        self.data_path = os.path.expandvars(self.data_path)

    def data_summary(self):
        """Data reduction

        The data processing, from photon data to light curves, is described <a href={link}>here</a>.   
        
        #### Specified source name: **{self.source_name}**

        {text}
        
        """

        #-------------------------------------------------------------------
        # check for data
        ok, text = self.check_data()
                
        otherdoc, link = self.docman.client('DataDescription')
       
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
            text+='not found locally -- checking for remote weight file...'

        t = self.get_weight_data()
        if t:
            os.makedirs(self.source_data_path, exist_ok=True)
            text += f'Weight file copied to '+t 
            return True, text
        return False, 'Failed to find it--please run get_weights'
            
        #     try:
        #         with sftp.Connection(self.server, self.username) as srv:
        #             srv.chdir(self.remote_data_path)
        #             flist =srv.listdir()
        #             if self.source_name in flist:
        #                 text += f'\nFound on remote, copying...:'
        #                 dest = os.path.join(self.data_path, self.source_name)
        # #                text += f' to {dest}'
        #                 if not os.path.isdir(dest):
        #                     os.makedirs(dest, exist_ok=True)
        #                 srv.chdir(self.source_name)
        #                 tocopy = srv.listdir()
        #                 for file in tocopy:
        #                     text += f'{file}, '
        #                     srv.get(file, dest+'/'+file)
        #             else:
        #                 text += 'Not found on remote.'
        #         return False, text
        #     except Exception as e:
        #         print(text+ f'\n*** Failed attempt to get remote files: {e.__repr__()}', file=sys.stderr)
        #         return False, text
                

        #     if not os.path.isdir(self.generation_path):
        #         text += f'\n*** No photon data for source "{self.source_name}" -- cannot generate on this machine.\n'
        #         text += f'Sources available here:\n'
        #         text += self.shell(f'ls {self.data_path}', monospace=False)
        #         #print(text, file = sys.stderr)
        #         return False, text
        # else:
        #     text += f'data  ok.'
        # return True, text

    def read_data(self):
        filename = self.source_data_path+'/time_data.pkl'
  
        if os.path.isfile(filename):
            self.timed_data = TimedDataX(filename)
            return self.already_generated(filename)
        else:
            os.makedirs(self.source_data_path, exist_ok=True)
            return self.generate(filename)

    def get_weight_data(self):
        fn = self.source_name.replace(' ','_').replace('+','p')+'_weights.pkl'
        if self.wsl:
            path = '/tmp/weight_files'
            # on a  local WSL machine
            import pysftp as sftp
            with sftp.Connection(self.server, self.username) as srv:
                srv.chdir(self.weight_path)
                files = srv.listdir()
                if fn not in files:
                    print(f'Did not find {fn} in {self.weight_path}')
                    return None
                os.makedirs(path, exist_ok=True)
                srv.get(fn, path+'/'+fn)

            return path
        else:
            # at SLAC
            files = glob.glob(self.weight_path)
            if fn not in files:
                    return None
            return self.weight_path
 
    def get_weights(self):
        """Weights

        This subsection summarizes data retrieved from the server {wd.server} for the pointlie skymodel {wd.skymodel}.

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
                weight_file=self.get_weight_data() ,
                parquet_root='$HOME/work/lat-data',
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
        <ul>
        <li>$t, tw$: time and width in JD units; 
        <li>_counts_: number of photons
        <li>_fext_: exposure factor, normaized to the average for a day
        <li>_flux_: fit value
        <li>_errors_: (lower, upper)
        <li>_limit_: 95% limit
        <li>_ts_: the TS value
        <li>poiss_pars: the three poisson-like parameters used to detemine flux, errors, limit, ts
        </ul>
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
    title: Photon Data Description
   
    """            
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

class DataFiles(DocPublisher):
    """
    title: Data Input Files 

    sections: basic_files source_selection example

    root: $HOME/work/lat-data
    ft2: ft2/*.fits
    gti: binned/*.fits
    effective_area: aeff

    verbose: 3
    example_source: Geminga

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        root = os.path.expandvars(self.root)
        assert os.path.isdir(root)
        self.data_file_pattern=os.path.join(root, self.gti)
        self.ft2_file_pattern=os.path.join(root, self.ft2)
        self.gti_file_pattern=os.path.join(root, self.gti)
        self.effective_area_path = os.path.join(root, self.effective_area)
        
        

    def basic_files(self):
        """Basic Data Files

        The various files used to construct the final product, a light curve for a given source,
        are located under a single root:
        {self.root}, which evaluates to {root} on this machine. Contents are:

        {ls_root}

        Running `{du_command}` shows the sizes: {du_result}

        The components are:

        * `photon_dataset`: The monthly files in HDF5 format containing each photon's time, position, 
        and energy, with the latter two binned for a combined 32 bands. These files are converted from
        pickled files generated by pointlike by `data_management.ParquetConversion`

        * `ft2`: Livetime from {ft2_size} monthly FT2-format files. The fields used are 
        `['LIVETIME','RA_SCZ','DEC_SCZ', 'RA_ZENITH','DEC_ZENITH']`. <br>These files are processed by
        data_management.TimedData._process_ft2.

        * `binned`: The pointlike-format binned data, used to obtain the GTI info.

        * `aeff`: Contains one or more effective area IRF files, in the "FB" format.

        * `sources`: Files specific to the data in a cone about a source position are here, described 
        in the next section

        """
        #-------------------------------------------------------------------
        import glob
        self.exroot = root = os.path.expandvars(self.root)
        ls_root = self.shell(f'ls -l {root}')
        du_command = f'(cd {root}; du  -d 1 .)'
        du_result = self.shell(f'{du_command}')
        ft2_size = len(glob.glob(root+'/ft2/*.fits'))
        self.publishme()

    def source_selection(self):
        """Source Selection
        """
        self.publishme()
    
    def example(self):
        """Example: Geminga

        This shows the processing for the source Geminga. We find its position using astropy, the code
        ```
        gal = SkyCoord.from_name(self.example_source).galactic
        l,b = (gal.l.value, gal.b.value)
        ```
        which results in $l,b$ = {l:.2f}, {b:.2f}

        We use `pointlike` to generate a file that allows us to assign a weight to each photon, based on
        its energy and band. For processing, this was copied to the folder `/tmp/weight_files`. For Geminga we have
        {weight_file}

        Then we use main.Main, generating the following:

        {text}


        """
        from astropy.coordinates import SkyCoord
        from lat_timing import Main
        source_name = self.example_source
        gal = SkyCoord.from_name(source_name).galactic
        l,b = (gal.l.value, gal.b.value)
        
        weight_file = self.shell(f'ls -l /tmp/weight_files/{source_name}*')

        with self.capture_print() as text:
            self.tdata = Main( name=source_name, 
                weight_file=f'/tmp/weight_files/',
                parquet_root='$HOME/work/lat-data',
                verbose=3,
                )

                
        self.publishme()

    def get_files(self, mjd_range=None):
        """ return lists of files to process
        """
        import glob
    
        data_files = sorted(glob.glob(os.path.expandvars(self.data_file_pattern)))
        assert len(data_files)>0, 'No files found using pattern {}'.format(self.data_file_pattern)
        if self.verbose>2:
            gbtotal = np.array([os.stat(filename).st_size for filename in data_files]).sum()/2**30
            print(f'Found {len(data_files)} monthly photon data files, with {gbtotal:.1f} GB total')

        ft2_files = sorted(glob.glob(os.path.expandvars(self.ft2_file_pattern)))
        gti_files = sorted(glob.glob(os.path.expandvars(self.gti_file_pattern)))
        assert len(ft2_files)>0 and len(gti_files)>0, 'Fail to find FT2 or GTI files'
        assert len(ft2_files)--len(gti_files), 'expect to find same number of FT2 and GTI files'

        if mjd_range is not None:
            mjd_range = np.array(mjd_range).clip(first_data, None)
            tlim =Time(mjd_range, format='mjd').datetime
            ylim,mlim = np.array([t.year for t in tlim])-2008, np.array([t.month for t in tlim])-1
            year_range = ylim.clip(0, len(gti_files)) + np.array([0,1])
            month_range = (ylim*12+mlim-8).clip(0,len(data_files)) + np.array([0,2]) #add 2 months?
            if self.verbose>1:
                print(f'From MJD range {mjd_range} select years {year_range}, months {month_range}')
        else:
            if self.verbose>2:
                print('Loading all found data')

        self.mjd_range = mjd_range # save for reference

        return (data_files if mjd_range is None else data_files[slice(*month_range)], 
                gti_files  if mjd_range is None else gti_files[slice(*year_range)], 
                ft2_files  if mjd_range is None else ft2_files[slice(*year_range)],
            )



