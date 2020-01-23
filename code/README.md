## Time analysis code

* Main
 * [main.py](https://github.com/tburnett/lat_timing/blob/master/code/main.py)<br>
Top level interface, via `class Main` 
 * [binner.py](https://github.com/tburnett/lat_timing/blob/master/code/binner.py) <br> 
 * [light_curve.py](https://github.com/tburnett/lat_timing/blob/master/code/light_curve.py)<br>  
 * [timed_data.py](https://github.com/tburnett/lat_timing/blob/master/code/timed_data.py)<br>
 * [data_management.py](https://github.com/tburnett/lat_timing/blob/master/code/data_management.py) <br>
 Load photon data as described below. Also load corresponding FT2 files and construct 
 * [weightman.py](https://github.com/tburnett/lat_timing/blob/master/code/weightman.py)<br>
 Manage weights--reads a file (described below) that has a map of the weight for each band and healpix nside=64 index.


* Time differencing
 * [EZPsearch.py](https://github.com/tburnett/lat_timing/blob/master/code/EZPsearch.py)  
 Copy of public code from UCSC for reference (python 2)
 * [time_diffs.py](https://github.com/tburnett/lat_timing/blob/master/code/time_diffs.py) 


* Kerr interface
 * [core_interface.py](https://github.com/tburnett/lat_timing/blob/master/code/core_interface.py)   
 * [core.py](https://github.com/tburnett/lat_timing/blob/master/code/core.py)    


* Utilities
 * [poisson.py](https://github.com/tburnett/lat_timing/blob/master/code/poisson.py)  
 * [keyword_options.py](https://github.com/tburnett/lat_timing/blob/master/code/keyword_options.py) 
 * [effective_area.py](https://github.com/tburnett/lat_timing/blob/master/code/)   
 

## Data location, formats

The 11-year dataset at the location `/nfs/farm/groups/glast/ `
### Photons<br>
I use `pointlike` code [pointlike/uw/data/timed_data](https://github.com/tburnett/pointlike/blob/master/python/uw/data/timed_data.py) to process FT1 files.
It applies the `time_record()` method of the class [`ConvertFT1`](https://github.com/tburnett/pointlike/blob/451c9e0fc5a888d771fe0274a3438594599dc442/python/uw/data/binned_data.py#L490).
This assigns events to 32 event `bands`, that is, 4/decade in energy and front/back, and position for likelihood analysis. Rather than generating bins, this code makes a record for each event with the band and position info. Each monthly FT1 file is processed independently. The use of float32 for the elapsed time within the FT1 file limits precision to a fraction of a second, adequate for all but pulsar timing.  
Its docstring:

```
        For selected events above 100 MeV, Create lists of the times and healpix ids
        (Reducing size from 20 to 9 bytes)
        returns:
            a recarray with dtype [('band', 'i1'), ('hpindex', '<i4'), ('time', '<f4')]
            where
                band:    energy band index*2 + 0,1 for Front/Back 
                hpindex: HEALPIx index for the nside 
                time:    the elapsed time in s from header value TSTART in the FT1 file
```
The current run, using notebook at `/nfs/farm/g/glast/g/catalog/pointlike/git/pointlike/python/uw/notebooks/make weights/create timed data.ipynb`
```
Sun Dec 1 17:19:22 PST 2019
132 monthly FT1 files found at /afs/slac/g/glast/groups/catalog/P8_P305/zmax105/*.fits
	 32 GB total
Writing time files to folder /afs/slac/g/glast/groups/catalog//pointlike/fermi/data/P8_P305/time_info
	overwrite=False
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
There are 132 timed data files, 4.8 GB tota
```

### Weights<br>
This also depends on `pointlike`. THe code is in [souurce_weights.py](https://github.com/tburnett/pointlike/blob/master/python/uw/like2/source_weights.py)

I am using the 10-year all-sky model uw9011. The notebook is here:

`/nfs/farm/g/glast/g/catalog/pointlike/git/pointlike/python/uw/notebooks/make weights/generate.ipynb`

It is run in the model folder `/nfs/farm/g/glast/g/catalog/pointlike/skymodels/P8_10years/uw9011`
with output for a given source.
```
Searching for name 504N-0010 ... Found in model:
Selecting ROI at (ra,dec)=(202.16,22.64)
84 total sources: 0 extended, 3 global
Found model source "504N-0010" within 0.000 deg of "504N-0010"
Generating pixels with weights for 8 energies
Using 102 nside=64 pixels.  distance range 0.57 to 5.20 deg
0 2 4 5 6 7 8 9 10 11 12 13 14 15 14 15 14 15 14 15 14 15 14 15 14 15 14 15 14 15
wrote file 504N-0010_weights.pkl
```

The file is a pickled dictionary with a `weight` key along with other info. The value is itself a dictionary with keys corresponding to 