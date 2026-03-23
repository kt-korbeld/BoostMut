# BoostMut
[![Powered by MDAnalysis](https://img.shields.io/badge/powered%20by-MDAnalysis-orange.svg?logoWidth=16&logo=data:image/x-icon;base64,AAABAAEAEBAAAAEAIAAoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJD+XwCY/fEAkf3uAJf97wGT/a+HfHaoiIWE7n9/f+6Hh4fvgICAjwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACT/yYAlP//AJ///wCg//8JjvOchXly1oaGhv+Ghob/j4+P/39/f3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJH8aQCY/8wAkv2kfY+elJ6al/yVlZX7iIiI8H9/f7h/f38UAAAAAAAAAAAAAAAAAAAAAAAAAAB/f38egYF/noqAebF8gYaagnx3oFpUUtZpaWr/WFhY8zo6OmT///8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgICAn46Ojv+Hh4b/jouJ/4iGhfcAAADnAAAA/wAAAP8AAADIAAAAAwCj/zIAnf2VAJD/PAAAAAAAAAAAAAAAAICAgNGHh4f/gICA/4SEhP+Xl5f/AwMD/wAAAP8AAAD/AAAA/wAAAB8Aov9/ALr//wCS/Z0AAAAAAAAAAAAAAACBgYGOjo6O/4mJif+Pj4//iYmJ/wAAAOAAAAD+AAAA/wAAAP8AAABhAP7+FgCi/38Axf4fAAAAAAAAAAAAAAAAiIiID4GBgYKCgoKogoB+fYSEgZhgYGDZXl5e/m9vb/9ISEjpEBAQxw8AAFQAAAAAAAAANQAAADcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAjo6Mb5iYmP+cnJz/jY2N95CQkO4pKSn/AAAA7gAAAP0AAAD7AAAAhgAAAAEAAAAAAAAAAACL/gsAkv2uAJX/QQAAAAB9fX3egoKC/4CAgP+NjY3/c3Nz+wAAAP8AAAD/AAAA/wAAAPUAAAAcAAAAAAAAAAAAnP4NAJL9rgCR/0YAAAAAfX19w4ODg/98fHz/i4uL/4qKivwAAAD/AAAA/wAAAP8AAAD1AAAAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALGxsVyqqqr/mpqa/6mpqf9KSUn/AAAA5QAAAPkAAAD5AAAAhQAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADkUFBSuZ2dn/3V1df8uLi7bAAAATgBGfyQAAAA2AAAAMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB0AAADoAAAA/wAAAP8AAAD/AAAAWgC3/2AAnv3eAJ/+dgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9AAAA/wAAAP8AAAD/AAAA/wAKDzEAnP3WAKn//wCS/OgAf/8MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIQAAANwAAADtAAAA7QAAAMAAABUMAJn9gwCe/e0Aj/2LAP//AQAAAAAAAAAA)](https://www.mdanalysis.org)

BoostMut is a python package to analyze high-throughput molecular dynamics simulations to investigate the effect of point mutations on stability.
Full details and benchmarking can be found in the [published paper]( https://doi.org/10.1002/pro.70334).
Since molecular dynamics simulations are costly, the simulations are expected to be short (50ps-1ns), using multiple (~5) runs to enhance sampling. Two of the analyses (the flexibility score of sidechains and hydrophobic surface exposure) require time and forcefield benchmarks against which the results are compared, but if these are ignored, BoostMut can be used with any ensemble of protein structures. The analysis and ranking produced by BoostMut is meant to serve as a secondary filter after an initial selection has been made using a primary stability predictor such a FoldX. 

## Setup
BoostMut is pip installable. To avoid conflicts with other packages, it is safest to do this within a virtual environment:
```
python -m venv boostmut-venv
source boostmut-venv/bin/activate
pip install BoostMut
```
BoostMut relies on the `freesasa` package to calculate solvent accessible surface. This package makes use of Cython to optimise calculations, which can cause problems in cases where C compilers are missing. After installing, BoostMut has a command line interface called `boostmut_run` for running calculations, and `boostmut_process` for processing the outputs. To get an overview of the available input flags, use:
```
boostmut_run -h
boostmut_process -h
```
## Input
BoostMut has the following advanced settings 
```
--inputdir        input directory containing subdirectories with trajectories for each mutation 
--output          name of the output .csv
--mutfile         file containing a list of mutations to analyze, if kept default, will analyze all mutations in input directory
--selection       reported selections per analysis
--time            length of the trajectory in picoseconds
--forcefield      forcefield with which the trajectory was run
--wtname          subdirectory for the wildtype
--mutname         regex the subdirectories for each mutation has to satisfy
--topname         regex each of the topology files has to satisfy 
--trajname        regex each of the trajectory files has to satify 
--bondsname       regex for seperate files with bondinfo if the topologies do not contain it 
--guessbonds      lets MDAnalysis guess bonds if topology is missing, making the calculations significantly slower (default: False)
--sasafile        name of custom file containing the benchmarks of residue sasa located in benchmarks. overrides time/forcefield 
--rmsffile        name of custom file containing the benchmarks for sidechain rmsf located in benchmarks. overrides time/forcefield 
--rangesur        range around the mutation used in the surrounding selection in Å (default: 8)
--rejecttraj      if set to True, rejects the trajectory with highest RMSD for each mutation (default: True)
--lastcheck       filename of the .csv from the last checkpoint from which to continue
--checkpoint      if set to True, saves the result after each mutation (default: True)
```
The input directory is set using `--inputdir`. BoostMut requires a directory with subdirectories for the wildtype and each of the mutations, each subdirectory in turn containing a set of trajectories and a topology. The input directory should have the following file structure:
```
input_directory
├── Subdir_D24K
├── ...
├── Subdir_T85V
└── Subdir_template
    ├── trajectory_1.xtc
    ├── ...
    ├── trajectory_5.xtc
    └── topology.tpr
```
By default BoostMut assumes `Subdir_template` as the directory name for the wildtype, `Subdir_[mutation]` as the directory name for each mutation, and the GROMACS specific `.xtc`  and `.tpr` files for the trajectories and topologies, but since BoostMut relies on MDAnalysis for parsing trajectories, it can handle all trajectory and topology formats [compatible with MDAnalsyis](https://userguide.mdanalysis.org/stable/formats/index.html). The expected pattern for the directory names, topologies and trajectories can be set by providing a [regex](https://regex101.com/). If the topology does not include bond information (such as .pdb files), bonds can be inferred by adding the `--guessbonds` flag, although this can significantly slow down calculation. Bonds can also be provided separately in a .tab file set using the `--bondsname` flag. If the default settings are sufficient, BoostMut can be run using:
```
boostmut_run --inputdir input_directory
```

## Analyses
BoostMut can analyze hydrogen bonding, RMSF of backbone and sidechains, hydrophobic surface exposure, and other structural checks. This can be done on three selections: the whole protein, 8Å surrounding a given mutation, or just the residue of the mutation. The final output returns a .csv with for each analysis and mutation the difference between mutant and wildtype. The analyses and the selections for each analysis can be customized in the command line. For example, if you want the surrounding and residue selections for hydrogen bonding, but only the whole protein selection for the other analyses, this can be specified with the -s flag with the desired analysis and selection divided by a colon:
```
boostmut_run --inputdir input_directory --selection hbse:sr c:p 
```
where the analyses are specified using:
* h : hydrogen bonding
* b : RMSF of backbone
* s : flexibility score of sidechains
* e : hydrophobic surface exposure
* c : other structural checks

and the selections are specified using:

* p : whole protein selection
* s : 8Å surrounding selection
* r : residue selection

By default, BoostMut assumes each trajectory is 50ps long and simulated with an amber forcefield. The analyses for the sidechain score and hydrophobic exposure rely on benchmark data for specific simulation lengths. The other analyses do not use benchmarks and therefore work the same regardless of timestep or forcefield. Benchmarks for different forcefields (available options are amber99, yamber3, charm27, and opls) in 50ps intervals for simulations up to 1000ps long are provided, and can be selected by setting the `--forcefield` and `--time` The appropriate benchmark files can be selected by providing the simulation length in the commandline interface:
```
boostmut_run --inputdir input_directory_500ps --time 500 --fforcefield opls
```
Alternatively, custom benchmark files can be set using `--sasafile`  For the expected hydrophobic exposure, and `--rmsffile` for the expected sidechain flexibilities per residue. For faster calculations (the main computational cost comes from re-aligning the structure for each side-chain when calculating the sidechain score) or simulations for which no forcefield benchmarks are available, the sidechain score and hydrophobic exposure can simply be left out of the analysis selection:
```
boostmut_run --inputdir input_directory --selection hb:sr c:p 
```
## Processsing
After the calculations have finished, the output can be processed with one of the tools in `boostmut_process` if needed. If the calculation of the mutations has been split up into separate parallel runs, the output has to be combined and rescaled. 
Combining can be done using `boostmut_process combine`. Rescaling the newly combined output file, or adding additional metrics can be done using `boostmut_process scale`. To obtain an easy human-readable excel version of the data, use `boostmut_process excel`. 





