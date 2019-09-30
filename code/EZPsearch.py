#! /usr/bin/env python
# version: v1
# date: 2013-07-08
# author: mario
# name: EZPsearch.py
# description:
#
#   The core technique is the time-differencing, described
#   in Atwood et al. 2006, applied by Ziegler et al. 2008
#   to EGRET data, and finally applied with great success
#   to LAT data in Abdo et al. 2009 (Science, 325, 840).
#   
#   The tool allows you to find a significant pulsation
#   in a time series of LAT events, by scanning in the spin
#   space of frequency and its first time derivative.
# 
#   The tool requires a barycentered event file, properly
#   filtered (either by "cookie-cutting" on energy and
#   ROI, or by selecting on Bayesian probability).
#
#   Several parameters and flags can be passed to the script.
#   The maximum frequency to search for, and the time window
#   for differencing have a key role. They determine the loss
#   in power due to truncation, the frequency resolution of
#   the search, and the size of the FFT:
#   N := FFT_size = 2 * Window_size * Max_frequency
#   The memory complexity of the FFTW algorithms, used for
#   the FFTs, scales as O(N), the CPU complexity as O(N*logN).
#
#   Other parameters fix the range in f1/f0 (by default, the
#   entire parameter space covered by known pulsars is scanned 
#   (that is from the value of the Crab=-1.3e-11 to 0)
#
# changelog:
#   * v0: initial revision
#   * v1: release for the 4th FAN workshop
__version__ = "2013-07-08"
# these modules come with the standard Python
import argparse
import sys, os
# reopen stdout to turn off the buffered output
try:
  sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
except:
  print 'buffered output not turned off'
# these modules are essential
try:
  from astropy.io import fits as pyfits
  import numpy as np
except:
  print "These Python modules are needed:"
  print "    pyfits,    numpy"
  print "Please verify that they are correctly installed"
  print "Check also your environment (eg. $PYTHONPATH)."
  print "Cannot load the required Python modules. Exit."
  sys.exit(1)
# FFTW is strongly recommended for performance
try:
  import pyfftwX
  use_fftw = True
except:
  print "Cannot load the Python module pyfftw."
  print "Use numpy.fft instead (the FFTs may be slower)."
  use_fftw = False

def main():
  try:
    # parse the options passed on the command line
    Options = ParseCommandLine()
    if not CheckConsistency(Options):
      raise Exception
  except IOError as e:
    print "I/O error(%d): %s. Exit." % (e.errno, e.strerror)
    sys.exit(2)
  except Exception: # what other errors can be raised?
    print "Cannot parse the command-line options. Exit."
    print "Try:", __file__, "-h"
    sys.exit(3)

  # read the event list from a LAT FT1 file
  [times, weights] = ReadEvents(
      Options.FT1_file, Options.weighted, Options.weight_column)
  if len(times) < 1:
    print "Cannot load the event list. Exit."
    sys.exit(4)
  # it should never happen...
  elif len(times) != len(weights):
    weights = [1] * len(times)
  epoch = (np.amax(times) + np.amin(times)) / 2.
  if not Options.quiet:
    print "Read %d events from '%s'" % (len(times), Options.FT1_file.name)
    print "Fix the epoch to MET %.0f" % (epoch)
    print "Compute FFTs of size %d" % \
        (FFT_Size(Options.window_size, Options.max_freq))
    print "The FFT resolution is %.5e Hz" % (1. / Options.window_size)

  # setup a basic search grid in -f1/f0
  p1_p0_step = GetP1_P0Step(
      times, Options.window_size, Options.max_freq)
  p1_p0_list = GetP1_P0List(
      p1_p0_step, Options.lower_p1_p0, Options.upper_p1_p0)
  print "Scan in p1/p0 between %.3e and %.3e s^-1" % \
      (Options.lower_p1_p0, Options.upper_p1_p0)
  print "The grid has %d points separated by %.3e s^-1\n" % \
      (len(p1_p0_list), p1_p0_step)

  step = 0
  OverallBest = [0, 0, 1]
  for p1_p0 in p1_p0_list:
    step += 1
    if not Options.quiet:
      print "Step in p1/p0 number %d out of %d: %.3e s^-1" %\
          (step, len(p1_p0_list), p1_p0)
    # correct for the frequency drift (careful: p1_p0=-f1/f0)
    new_times = TimeWarp(times, p1_p0, epoch)
    # work out the binned series of time differences
    time_differences = TimeDiffs(
        new_times, weights, Options.window_size, Options.max_freq)
    # Fourier transform the series of time differences
    if use_fftw:
      power_spectrum = FFTW_Transform(
          time_differences, Options.window_size, Options.max_freq)
    else:
      power_spectrum = FFT_Transform(
          time_differences, Options.window_size, Options.max_freq)
    # pick the pulsar candidate with highest Fourier power 
    [freq, p_value] = ExtractBestCandidate(
        power_spectrum, Options.min_freq, Options.max_freq)
    if not Options.quiet:
      DisplayCandidate([freq, p1_p0, p_value])
    # store the best pulsar candidate
    if p_value <= OverallBest[2]:
      OverallBest = [freq, p1_p0, p_value]

  # show a summary of the search
  print "\nScan in -f1/f0 completed after %d steps" % (len(p1_p0_list))
  DisplayCandidate(OverallBest, best=True)

def ParseCommandLine():
  """ParseCommandLine()"""
  """Read the arguments passed on the command line"""
  """to the script. Only the data file is mandatory"""
  """while the defaults for the other parameters"""
  """agree with the values in Ziegler et al 2008."""
  Parser = argparse.ArgumentParser(
      description='Run a pulsar blind search on LAT data')
  Parser.add_argument('FT1_file', type=argparse.FileType('r'))
  # These defaults agree with Ziegler et al 2008
  Parser.add_argument('--window_size', '-s', type=float, \
      default=524288.0, help='maximum time diff in sec [524288]')
  Parser.add_argument('--max_freq', '-f', type=float, \
      default=64.0, help='maximum frequency in Hz [64]')
  Parser.add_argument('--min_freq', '-m', type=float, \
      default=.5, help='minimum frequency in Hz [0.5]')
  # XXX can argparse deal with negative values? how?
  Parser.add_argument('--lower_p1_p0', '-l', type=float, \
      default=0.0, help='lower bound of p1/p0 in s^-1 [0]')
  Parser.add_argument('--upper_p1_p0', '-u', type=float, \
      default=1.3e-11, help='upper bound of p1/p0 in s^-1 [1.3e-11 (Crab)]')
  # Weghts were not originally included in the blind search
  Parser.add_argument('--weighted', '-w', action='store_true', \
      default=False, help='turn on the usage of weights')
  Parser.add_argument('--weight_column', '-c', \
      default='PROB', help='column of the FT1 file storing the weights [PROB]')
  Parser.add_argument('--quiet', '-q', action='store_true', \
      default=False, help='turn off most of the messages')
  Parser.add_argument('--version', '-v', action='version', version='%(prog)s 1.0')
  return Parser.parse_args()

def CheckConsistency(Options):
  """CheckConsistency()"""
  """Verify that the parameter values, either set"""
  """on the command line or taken from the defaults"""
  """are compatible."""
  if Options.lower_p1_p0 > Options.upper_p1_p0:
    print "Invalid range for -f1/f0: [%.3e, %.3e]" % \
        (Options.lower_p1_p0, Options.upper_p1_p0)
    return False
  if Options.min_freq >= Options.max_freq:
    print "Invalid range for f0: [%.3e, %.3e]" % \
        (Options.min_freq, Options.max_freq)
    return False
  if Options.min_freq * Options.window_size <= 1.0:
    print "Window size [%s s] too small for f0=%g Hz" % \
        (Options.window_size, Options.min_freq)
    return False
  return True

def FFT_Size(window_size, max_freq):
  """FFT_Size()"""
  """The size of the FFT is twice the product of"""
  """the window size times the maximum frequency."""
  return 2 * int(np.floor(window_size * max_freq))

def ReadEvents(FT1_file, weighted = False, weight_column = "PROB"):
  """ReadEvents()"""
  """Read times and weights from a LAT FT1 file"""
  """Mind that the file must have been previously"""
  """filtered and barycentered with gtselect and"""
  """gtbary. Also, the weight column must exist"""
  """already, if you want to run a weighted search"""
  # exit smoothly if the event file cannot be read
  try:
    EventList = pyfits.getdata(FT1_file.name)
    times = EventList.field("TIME")
  except:
    print "Cannot read the event list from '%s'" % (FT1_file.name)
    return [[],[]]

  # try to read the weight column
  if weighted:
    try:
      weights = EventList.field(weight_column)
    # in case of problems don't use the weights 
    except:
      print "Cannot load the weight column '%s'" % (weight_column)
      weighted = False
  if not weighted:
    weights = [1] * len(times)
  return [times, weights]

def TimeDiffs(times, weights, window_size = 524288, max_freq = 64):
  """TimeDiffs()"""
  """Extract the binned series of time differences"""
  """The argument max_freq determines the bin size"""
  """as time_resol = 1 / (2 * max_freq) """
  """This together with window_size fixes the size"""
  """of the returned array of time differences"""
  # FFT sampling time
  time_resol = .5 / max_freq
  # directly bin the time differences
  time_diffs = [0] * FFT_Size(window_size, max_freq)
  for i1 in range(len(times) - 1):
    t1 = times[i1]
    for i2 in range(i1 + 1, len(times)):
      t2 = times[i2]
      # limit the size of the time differences
      if t2 - t1 > window_size:
        break
      # determine the frequency bin in the array
      freq_bin = int(np.floor((t2 - t1) / time_resol))
      # combine the weights appropriately
      time_diffs[freq_bin] += weights[i1] * weights[i2]
  return time_diffs

def GetP1_P0Step(times, window_size = 524288, max_freq = 64):
  """GetP1_P0Step()"""
  """Determine the grid step in the P1/P0 parameter space"""
  """such that the maximum tolerated frequency drift over"""
  """the full time span covered by the data is smaller than"""
  """the FFT resolution (see eq.3 in Ziegler et al 2008)"""
  """for the largest frequency considered in the search"""
  time_span = np.amax(times) - np.amin(times)
  FFT_resol = 1. / window_size
  # this value is somewhat arbitrary: a finer grid provides
  # a better sensitivity, but is more time-consuming; a coarser
  # grid is clearly faster but you risk of missing the pulsar 
  f1_tolerance = 1. * FFT_resol / time_span
  # at least one point in the grid is within 1/2 the grid
  # step from the correct value of the parameter (p1/p0) 
  return 2. * f1_tolerance / max_freq

def GetP1_P0List(p1_p0_step, lower_p1_p0=0., upper_p1_p0=1.3e-11):
  """GetP1_P0List()"""
  """Work out the grid, given the step and the boundaries"""
  # add one last step, even if it falls outside of the range
  return np.arange(lower_p1_p0, upper_p1_p0+p1_p0_step, p1_p0_step)
 
def TimeWarp(times, p1_p0, epoch):
  """TimeWarp()"""
  """Stretch a time series in order to compensate for a"""
  """steady frequency drift. This way, the new time series"""
  """is periodic and can be searched with standard FFTs."""
  """This time transform is described in Ransom et al. 2001"""
  """See also eq 1 in Ziegler et al. 2008"""
  # the minus sign enters because p1_p0 is p1/p0=-f1/f0
  return [T - .5*p1_p0*(T-epoch)**2 for T in times]

def FFTW_Transform(time_differences, window_size, max_freq):
  """FFTW_Transform()"""
  """Prepare the FFTW engine, allocate memory, and compute"""
  """the FFT, returning the normalized Fourier power vector"""
  """For background on FFTW see: www.fftw.org"""
  """For the python wrapper: hgomersall.github.io/pyFFTW/"""
  FFT_size = FFT_Size(window_size, max_freq)
  alignment = pyfftw.simd_alignment
  # this is tricky: it is needed to get the correct memory alignment for fftw
  input_array = pyfftw.n_byte_align_empty(FFT_size, alignment, dtype='float32')
  output_array = pyfftw.n_byte_align_empty(FFT_size//2+1, alignment, dtype='complex64')

  # create the FFT object, BEFORE actually loading the data!!!!
  fft_object = pyfftw.FFTW(input_array, output_array, threads=1)

  # load the actual input into the allocated memory
  input_array[:] = time_differences
  # this normalization grants that, if the input array is Poisson distributed,
  # the Fourier power follows a chi2 distribution with 2 degrees of freedom
  # unfortunately the time differences are NOT Poisson distributed...
  norm = np.sum(np.absolute(input_array)/2.0, dtype=np.float32)
  # FFTW.__Call__ automatically executes the FFT and returns the output array
  output_array = fft_object()
  # return the normalized Fourier power
  return np.square(np.absolute(output_array)) / norm

def FFT_Transform(time_differences, window_size, max_freq):
  """FFT_Transform()"""
  """Compute the FFT of the time differences, using numpy,"""
  """and return the normalized Fourier power vector"""
  # this normalization grants that, if the input array is Poisson distributed,
  # the Fourier power follows a chi2 distribution with 2 degrees of freedom
  # unfortunately the time differences are NOT Poisson distributed...
  norm = np.sum(np.absolute(time_differences)/2.0, dtype=np.float32)
  # numpy offers an interface optimized for real FFTs
  output_array = np.fft.rfft(time_differences)
  # return the normalized Fourier power
  return np.square(np.absolute(output_array)) / norm

def ExtractBestCandidate(power_spectrum, min_freq, max_freq):
  """ExtractBestCandidate()"""
  """Pick the candidate frequency with the largest Fourier"""
  """power, within the prescribed frequency range."""
  """Return the pair of its frequency and the P-value of"""
  """observing such a power as a statistical fluctuation."""
  """For a discussion, see Sec 4 of Ziegler et al. 2008"""
  # This value is roughly correct, though FFT_resol := 1/window_size 
  FFT_resol = float(max_freq) / (len(power_spectrum) - 1.0)
  # Ignore peaks at the lowest frequencies, in order to avoid red noise
  min_index = int(np.floor(float(min_freq) / FFT_resol))
  peak_index = min_index + np.argmax(power_spectrum[min_index:])
  # We nned this operation of CPU complexity NlogN to interpret the power
  sorted_power = np.sort(power_spectrum[min_index:], kind='heapsort')[::-1]
  # Work out the asymptotic cumulative distribution of the power values
  [slope, constant] = FitExponentialTail(sorted_power)
  # Apply the asymptotic results to convert the power into a P-value
  P_value = PowerToPValue(sorted_power[0], slope, constant) 
  return [FFT_resol * peak_index, P_value]

def FitExponentialTail(sorted_array):
  """FitExponentialTail()"""
  """Analyze the probability distribution of values in an"""
  """array and fit the tail with an exponential function."""
  """The array is assumed as already sorted (decreasing)."""
  # We define the tail through an emprical approximation
  if len(sorted_array) > 2000000:
    start_index = 200
    end_index = 20000
  else:
    start_index = int(len(sorted_array) / 10000)
    end_index = int(len(sorted_array) / 100)
  # consider only the fit range and assume an exponential shape
  X_values = sorted_array[start_index:end_index]
  Y_values = np.log(np.arange(start_index+1, end_index+1))
  # this is a bit overkilling: a line is a polynomial of order 1
  return np.polyfit(X_values, Y_values, 1)

def PowerToPValue(power, slope, constant):
  """PowerToPValue()"""
  """Apply the asymptotic cumulative distribution of the"""
  """power under the null hypothesis of no pulsations, to"""
  """estimate the probability of getting by chance values"""
  """at least as extreme as the one observed (P-value)"""
  # this value accounts already for the trials due to FFT bins
  # it comes from an empirical fit, and is very approximative
  effective_power = power - np.sqrt(power)
  return np.amin([np.exp(constant + slope*effective_power), 1])

def DisplayCandidate(candidate, best=False):
  """DisplayCandidate()"""
  """Print the basic information about a pulsar candidate"""
  if best:
    print "\nThe best pulsar candidate is:"
  # the second entry in the candidate is the value of p1/p0=-f1/f0
  Fdot = -1.*candidate[1]*candidate[0]
  print "F0=%.8f F1=%.3e P-Value=%.2e" % (candidate[0], Fdot, candidate[2])
  if best:
    print "Characteristic age=%.2e years" % (3.1688e-08 / candidate[1])
    print "Edot=I45*%.2e erg/s\n" % (-3.9478e46 * Fdot * candidate[0])
    

if __name__ == '__main__':
  main()
