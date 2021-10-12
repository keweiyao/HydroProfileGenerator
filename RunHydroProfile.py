#!/usr/bin/env python3

# python modules
import argparse
from contextlib import contextmanager
from itertools import chain, groupby, repeat
import logging
import math
import os
import subprocess
import sys
import tempfile

# third party
import numpy as np
import h5py
from scipy.interpolate import interp1d
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.interpolation import rotate, shift

# hic-eventgen specific modules
import freestream

def run_cmd(*args):
    """
    Run and log a subprocess.
    """
    cmd = ' '.join(args)
    logging.info('running command: %s', cmd)

    try:
        proc = subprocess.run(
            cmd.split(), check=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        print(proc.stdout)
        print(proc.stderr)
    except subprocess.CalledProcessError as e:
        logging.error(
            'command failed with status %d:\n%s',
            e.returncode, e.output.strip('\n')
        )
        raise
    else:
        logging.debug(
            'command completed successfully:\n%s',
            proc.stdout.strip('\n')
        )
        return proc

def read_text_file(filename):
    """
    Read a text file into a nested list of bytes objects,
    skipping comment lines (#).
    """
    with open(filename, 'rb') as f:
        return [l.split() for l in f if not l.startswith(b'#')]

class Parser(argparse.ArgumentParser):
    """
    ArgumentParser that parses files with 'key = value' lines.
    """
    def __init__(self, *args, fromfile_prefix_chars='@', **kwargs):
        super().__init__(
            *args, fromfile_prefix_chars=fromfile_prefix_chars, **kwargs
        )

    def convert_arg_line_to_args(self, arg_line):
        # split each line on = and prepend prefix chars to first arg so it is
        # parsed as a long option
        args = [i.strip() for i in arg_line.split('=', maxsplit=1)]
        args[0] = 2*self.prefix_chars[0] + args[0]
        return args


parser = Parser(
    usage=''.join('\n  %(prog)s ' + i for i in [
        '[options] <results_file>',
        '-h | --help',
    ]),
    description='''
''',
    formatter_class=argparse.RawDescriptionHelpFormatter
)

parser.add_argument(
    '--working-dir', type=os.path.abspath, metavar='PATH', default="./",
    help='specfiy a working directory, default is ./'
)
parser.add_argument(
    '--nevents', type=int, metavar='INT',
    help='number of events to run (default: run until interrupted)'
)
parser.add_argument(
    '--avg-ic', default='off', metavar='VAR',
    help='if on, average a bunch of initial condition'
)
parser.add_argument(
    '--avg-n', type=np.int, default=100, metavar='INT',
    help='Number of events to be averaged, only used if --avg-ic is on'
)
parser.add_argument(
    '--centrality-bins', type=str, default='0 100', metavar='STR',
    help='centrality bin, only works for avg-ic=on'
)
parser.add_argument(
    '--Nevents', type=int, default=100, metavar='INT',
    help='number of ebe run_events'
)
parser.add_argument(
    '--proj', type=str, default="Pb", metavar='STR',
    help='projectile'
)
parser.add_argument(
    '--targ', type=str, default="Pb", metavar='STR',
    help='target'
)
parser.add_argument(
    '--trento-args', default='', metavar='ARGS',
    help="arguments passed to trento (default: '%(default)s')"
)
parser.add_argument(
    '--tau-fs', type=float, default=.5, metavar='FLOAT',
    help='free streaming time [fm] (default: %(default)s fm)'
)
parser.add_argument(
    '--hydro-args', default='', metavar='ARGS',
    help='arguments passed to osu-hydro (default: empty)'
)
parser.add_argument(
    '--Tswitch', type=float, default=.150, metavar='FLOAT',
    help='particlization temperature [GeV] (default: %(default).3f GeV)'
)

# fully specify numeric data types, including endianness and size, to
# ensure consistency across all machines
float_t = '<f8'
int_t = '<i8'
complex_t = '<c16'

def initial_conditions(initial_file='ic.h5', avg='off', grid_step=0.1, grid_max=15.0):
    """
    Handles initial initial_conditions
    If avg='on', it will first run TRENTo with the given setting for 1000 times.
    Then, each event is rotated so that the participant plane phi2 aligns to
    produce a more realistic event averaging. The binary collision density is
    performed similar as the initial energy density grid.
    """
    def average_ic(cenL, cenH):
        # decide the centarity range
        mult = []
        with h5py.File(initial_file, 'r') as f:
            for eve in f.values():
                mult.append(eve['matter_density'][()].sum())
        # quantile of multiplcity at the initial condition level
        mult_L = np.quantile(mult, 1.-cenH/100.)
        mult_H = np.quantile(mult, 1.-cenL/100.)
        Ncount = 0
        avgNpart = 0
        avg_b = 0
        with h5py.File(initial_file, 'r') as f:
            densityavg = np.zeros_like(f['event_0/matter_density'][()])
            Ncollavg = np.zeros_like(f['event_0/Ncoll_density'][()])
            dxy = f['event_0'].attrs['dxy']
            Neve = len(f.values())
            for eve in f.values():
                # step1, center the event
                Npart = eve.attrs['npart']
                b = eve.attrs['b']
                NL = int(eve.attrs['Nx']/2)
                density = eve['matter_density'][()]
                cenkey = density.sum()
                if cenkey<mult_L or mult_H<cenkey:
                    continue
                else:
                    Ncount += 1
                    avgNpart += Npart
                    avg_b += b
                comxy = -np.array(center_of_mass(density))+np.array([NL, NL])
                density = shift(density, comxy)
                Ncoll = shift(eve['Ncoll_density'][()], comxy)
                # step2, rotate the event to align psi2
                psi2 = eve.attrs['psi2']
                imag_psi2 = psi2*180./np.pi + (90. if psi2<0 else -90.)
                densityavg += rotate(density, angle=imag_psi2, reshape=False)
                Ncollavg += rotate(Ncoll, angle=imag_psi2, reshape=False)
            # step3 take average
            densityavg /= Ncount
            Ncollavg /= Ncount
            avgNpart /= Ncount
            avg_b /= Ncount
        logging.info("averaging over {} events that satisfy the centrlaity cut out of {} events".format(Ncount, Neve))
        logging.info("Centrality {}-{} has an average Npart = {}, avg b = {} [fm]".format(cenL, cenH, avgNpart, avg_b))
        # rewrite the initial.hdf file with average ic
        os.remove(initial_file)

        with h5py.File(initial_file, 'w') as f:
            gp = f.create_group('event_0')
            gp.create_dataset('matter_density', data=densityavg)
            gp.create_dataset('Ncoll_density', data=Ncollavg)
            gp.attrs.create('Nx', densityavg.shape[1])
            gp.attrs.create('Ny', densityavg.shape[0])
            gp.attrs.create('dxy', dxy)

    try:
        os.remove(initial_file)
    except FileNotFoundError:
        pass
    cenL, cenH = 0, 100

    if avg == 'on':
        cenL, cenH = [float(it) for it in args.centrality_bins.split()]
        logging.info("centrality selection {} - {} %".format(cenL, cenH))
        logging.info("averaged initial condition mode, could take a while")
    else:
        logging.info("Minimum biased events")
    run_cmd(
        'trento',
        '{} {}'.format(args.proj, args.targ),
        '--number-events {}'.format(1 if avg=='off' else args.avg_n),
        '--grid-step {} --grid-max {}'.format(grid_step, grid_max),
        '--output', initial_file,
        args.trento_args,
    )
    if avg == 'on':
        average_ic(cenL, cenH)

    with h5py.File(initial_file, 'r') as f:
        ic = np.array(f['event_0/matter_density'])
        nb = np.array(f['event_0/Ncoll_density'])
    return ic, nb

def save_fs_with_hydro(ic, grid_max):
    """
    The freestream energy density will be saved at the begining of the hydro file
    Before:
    Frame0000, Frame0001, ..., FrameXXXX of hydrodynamics
    After:
    Frame0000, Frame0001, ..., FrameYYYY, FrameYYYY+1, ..., FrameXXXX+YYYY
    |<----------Freestreaming---------->|<-------Hydrodynamics---------->|
    """
    # use same grid settings as hydro output
    with h5py.File('JetData.h5','a') as f:
        taufs = f['Event'].attrs['Tau0'][0]
        dtau = f['Event'].attrs['dTau'][0]
        dxy = f['Event'].attrs['DX'][0]
        ls = f['Event'].attrs['XH'][0]
        n = 2*ls + 1
        # [tau0, tau0+dtau, tau0+2*dtau, ..., taufs - dtau] + hydro steps...
        nsteps = int(taufs/dtau)
        tau0 = taufs-dtau*nsteps
        if tau0 < 1e-2: # if tau0 too small, skip the first step
            tau0 += dtau
            nsteps -= 1
        taus = np.linspace(tau0, taufs-dtau, nsteps)
        # First, rename hydro frames and leave the first few name slots to FS
        event_gp = f['Event']
        for i in range(len(event_gp.keys()))[::-1]:
            old_name = 'Frame_{:04d}'.format(i)
            new_name = 'Frame_{:04d}'.format(i+nsteps)
            event_gp.move(old_name, new_name)
        # Second, overwrite tau0 with FS starting time, and save taufs where
        # FS and hydro is separated
        event_gp.attrs.create('Tau0', [tau0])
        event_gp.attrs.create('TauFS', [taufs])
        # Thrid, fill the first few steps with Freestreaming results
        for itau, tau in enumerate(taus):
            frame = event_gp.create_group('Frame_{:04d}'.format(itau))
            fs = freestream.FreeStreamer(ic, grid_max, tau)
            for fmt, data, arglist in [
                ('e', fs.energy_density, [()]),
                ('V{}', fs.flow_velocity, [(1,), (2,)]),
                ('Pi{}{}', fs.shear_tensor, [(0,0), (0,1), (0,2),
                                                    (1,1), (1,2),
                                                            (2,2)] ),
                ]:
                for a in arglist:
                    X = data(*a).T # to get the correct x-y with vishnew
                    if fmt == 'V{}': # Convert u1, u2 to v1, v2
                        X = X/data(0).T
                    diff = X.shape[0] - n
                    start = int(abs(diff)/2)
                    if diff > 0:
                        # original grid is larger -> cut out middle square
                        s = slice(start, start + n)
                        X = X[s, s]
                    elif diff < 0:
                        # original grid is smaller
                        #  -> create new array and place original grid in middle
                        Xn = np.zeros((n, n))
                        s = slice(start, start + X.shape[0])
                        Xn[s, s] = X
                        X = Xn
                    if fmt == 'V{}':
                        Comp = {1:'x', 2:'y'}
                        frame.create_dataset(fmt.format(Comp[a[0]]), data=X)
                    if fmt == 'e':
                        frame.create_dataset(fmt.format(*a), data=X)
                        frame.create_dataset('P', data=X/3.)
                        frame.create_dataset('BulkPi', data=X*0.)
                        prefactor = 1.0/15.62687/5.068**3
                        frame.create_dataset('Temp', data=(X*prefactor)**0.25)
                        s = (X + frame['P'][()])/(frame['Temp'][()]+1e-14)
                        frame.create_dataset('s', data=s)
                    if fmt == 'Pi{}{}':
                        frame.create_dataset(fmt.format(*a), data=X)
            pi33 = -(frame['Pi00'][()] + frame['Pi11'][()] \
                                            + frame['Pi22'][()])
            frame.create_dataset('Pi33', data=pi33)
            pi3Z = np.zeros_like(pi33)
            frame.create_dataset('Pi03', data=pi3Z)
            frame.create_dataset('Pi13', data=pi3Z)
            frame.create_dataset('Pi23', data=pi3Z)

def run_fs_and_hydro(initial_density, args, hydro_grid_max, dxy):
    """
    Run FreeStream + Hydrodynamics
    """
    # first, free-stream
    fs = freestream.FreeStreamer(initial_density, hydro_grid_max, args.tau_fs)
    ls = math.ceil(hydro_grid_max/dxy)  # the osu-hydro "ls" parameter
    n = 2*ls + 1  # actual number of grid cells
    for fmt, f, arglist in [
            ('ed', fs.energy_density, [()]),
            ('u{}', fs.flow_velocity, [(1,), (2,)]),
            ('pi{}{}', fs.shear_tensor, [(1, 1), (1, 2), (2, 2)]),
    ]:
        for a in arglist:
            X = f(*a)
            diff = X.shape[0] - n
            start = int(abs(diff)/2)

            if diff > 0:
                # original grid is larger -> cut out middle square
                s = slice(start, start + n)
                X = X[s, s]
            elif diff < 0:
                # original grid is smaller
                #  -> create new array and place original grid in middle
                Xn = np.zeros((n, n))
                s = slice(start, start + X.shape[0])
                Xn[s, s] = X
                X = Xn
            X.tofile(fmt.format(*a) + '.dat')
    dt = dxy*0.25
    run_cmd(
        'osu-hydro',
        't0={} dt={} dxy={} nls={}'.format(args.tau_fs, dt, dxy, ls),
        args.hydro_args
    )
    logging.info("Save free streaming history with hydro histroy")
    save_fs_with_hydro(initial_density, hydro_grid_max)



def run_events(args):
    density, ncoll = initial_conditions(initial_file='ic.h5', avg=args.avg_ic, grid_step=0.1, grid_max=15.0)
    run_fs_and_hydro(density, args, hydro_grid_max=15.0, dxy=0.1)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    args = parser.parse_args()
    if not os.path.exists(args.working_dir):
        os.makedirs(args.working_dir)
    os.chdir(args.working_dir)
    run_events(args)
