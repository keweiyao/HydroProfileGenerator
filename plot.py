#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

plt.figure(figsize=(10,4))
with h5py.File(sys.argv[1],'r') as f:
     frames = f['Event'].keys()
     attrs = f['Event'].attrs
     dxy = attrs['DX'][0]
     tau0 = attrs['Tau0'][0]
     dtau = attrs['dTau'][0]
     taufs = attrs['TauFS'][0]
     halfL = attrs['XH'][0]*dxy
     Nxy = attrs['XH'][0]*2+1
     x = np.linspace(-halfL, halfL, Nxy)
     X, Y = np.meshgrid(x,x)
     X = X.T.flatten()
     Y = Y.T.flatten()
     extent = -halfL,halfL,-halfL,halfL
     for i, it in enumerate(frames):
         tau = tau0 + i*dtau
         plt.clf()
         T = f['Event'][it]['Temp'][()]
         bounds = (0.149<T) & (T<0.151)
         for j, (name, unit) in enumerate(zip(['Temp', 'Vx', 'Vy'],
                                              [' [GeV]', '', ''])
                                         ):
             field = f['Event'][it][name][()]
             plt.subplot(1,3,j+1)
             plt.imshow(np.flipud(field.T), extent=extent)
             plt.title(name+unit)
             plt.xlabel(r"$x$ [fm]")
             plt.ylabel(r"$y$ [fm]")
             plt.scatter(X, Y, bounds.flatten(), color='gray', alpha=.5)
         label = 'free-stream' if tau<=taufs else 'hydrodynamics'
         plt.suptitle(r"$\tau={:1.2f}$ [fm/c], {:s}".format(tau, label))
         plt.tight_layout(True)
         plt.pause(.1)
plt.show()
