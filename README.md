# HydroProfileGenerator
A minimal version hydro profile generator for jet evolution. THis is based on hic-eventgen.

Dependences: `gfortran`, `c++11` and above, `boost`, `gsl`, `hdf5`, `python3`, `numpy`, `matplotlib`, `h5py`, `scipy` 

```bash
   mkdir bulkmodel && cd bulkmodel

   pip install frzout
   pip install freestream
   # or you can install from source

   git clone -b Ncoll-density https://github.com/keweiyao/trento3d
   cd trento3d && mkdir build
   cd build && cmake ..
   make install
   cd ..
   # this will install to $HOME/.local/bin by default

   git clone -b hydro_profile https://github.com/keweiyao/vishnew
   cd vishnew && mkdir build
   cd build && cmake ..
   make install
   cd ..
   # this will install to $HOME/.local/bin by default
```

Set the path `export PATH=$PATH/$HOME/.local/bin` and `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/lib` and `export XDG_DATA_HOME=$HOME/.local/share`. You can put these settings in your `~/.bashrc` and then `source ~/.bashrc`.


Finally, clone this repo
```bash
   git clone https://github.com/keweiyao/HydroProfileGenerator
   cd HydroProfileGenerator
```
Now, try 
```bash
   ./RunHydroProfile.py --working-dir=./Run @PbPb5020_config.dat
```

And then 
```
   ./plot.py Run/JetData.h5 
```
