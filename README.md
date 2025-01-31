# BoostMut
[![Powered by MDAnalysis](https://img.shields.io/badge/powered%20by-MDAnalysis-orange.svg?logoWidth=16&logo=data:image/x-icon;base64,AAABAAEAEBAAAAEAIAAoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJD+XwCY/fEAkf3uAJf97wGT/a+HfHaoiIWE7n9/f+6Hh4fvgICAjwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACT/yYAlP//AJ///wCg//8JjvOchXly1oaGhv+Ghob/j4+P/39/f3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJH8aQCY/8wAkv2kfY+elJ6al/yVlZX7iIiI8H9/f7h/f38UAAAAAAAAAAAAAAAAAAAAAAAAAAB/f38egYF/noqAebF8gYaagnx3oFpUUtZpaWr/WFhY8zo6OmT///8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgICAn46Ojv+Hh4b/jouJ/4iGhfcAAADnAAAA/wAAAP8AAADIAAAAAwCj/zIAnf2VAJD/PAAAAAAAAAAAAAAAAICAgNGHh4f/gICA/4SEhP+Xl5f/AwMD/wAAAP8AAAD/AAAA/wAAAB8Aov9/ALr//wCS/Z0AAAAAAAAAAAAAAACBgYGOjo6O/4mJif+Pj4//iYmJ/wAAAOAAAAD+AAAA/wAAAP8AAABhAP7+FgCi/38Axf4fAAAAAAAAAAAAAAAAiIiID4GBgYKCgoKogoB+fYSEgZhgYGDZXl5e/m9vb/9ISEjpEBAQxw8AAFQAAAAAAAAANQAAADcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAjo6Mb5iYmP+cnJz/jY2N95CQkO4pKSn/AAAA7gAAAP0AAAD7AAAAhgAAAAEAAAAAAAAAAACL/gsAkv2uAJX/QQAAAAB9fX3egoKC/4CAgP+NjY3/c3Nz+wAAAP8AAAD/AAAA/wAAAPUAAAAcAAAAAAAAAAAAnP4NAJL9rgCR/0YAAAAAfX19w4ODg/98fHz/i4uL/4qKivwAAAD/AAAA/wAAAP8AAAD1AAAAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALGxsVyqqqr/mpqa/6mpqf9KSUn/AAAA5QAAAPkAAAD5AAAAhQAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADkUFBSuZ2dn/3V1df8uLi7bAAAATgBGfyQAAAA2AAAAMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB0AAADoAAAA/wAAAP8AAAD/AAAAWgC3/2AAnv3eAJ/+dgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9AAAA/wAAAP8AAAD/AAAA/wAKDzEAnP3WAKn//wCS/OgAf/8MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIQAAANwAAADtAAAA7QAAAMAAABUMAJn9gwCe/e0Aj/2LAP//AQAAAAAAAAAA)](https://www.mdanalysis.org)


BoostMut is a python package to analyze high-throughput molecular dynamics simulations to investigate the effect of point mutations on stability.
Since molecular dynamics simulations are costly, the simulations are short (50ps-1ns) and their analysis by BoostMut are meant to serve as a secondary filter after an initial selection has been made using a primary predictor such a FoldX. As input, BoostMut requires a directory with subdirectories for the wildtype and each of the mutations, each subdirectory in turn containing a set of short trajectories and a topology:
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
After installing, BoostMut is run in the command line.  using:
```
boostmut_run -h
```
BoostMut can analyze hydrogen bonding, RMSF of backbone and sidechains, hydrophobic surface exposure, and other structural checks. This can be done on three selections: the whole protein, 8Å surrounding a given mutation, or just the residue of the mutation. The final output returns a .csv with for each analysis and mutation the difference between mutant and wildtype. The analyses and the selections for each analysis can be customized in the command line. For example, if you want the surrounding and residue selections for hydrogen bonding, but only the whole protein selection for the other analyses, this can be specified with the -a flag for specifying the analyses, and the -s flag specifying each selection given as output per analysis:
```
boostmut_run -i input_directory -a hrsc -s h:sr, rsc:p 
```
where the analyses are specified using:
* h : hydrogen bonding
* r : RMSF of backbone and sidechains
* s : hydrophobic surface exposure
* c : other structural checks

and the selections are specified using:

* p : whole protein selection
* s : 8Å surrounding selection
* r : residue selection

By default, BoostMut assumes each trajectory is 50ps long. The analyses for the sidechain RMSF and hydrophobic exposure rely on benchmark data for specific simulation lengths, and so will give erroneous results if the wrong timestep is given. The other analyses do not use benchmarks and therefore work the same regardless of timestep. In case longer simulations are used, benchmarks for each 50ps interval for simulations up to 1000ps long are provided. The appropriate benchmark files can be selected by providing the simulation length in the commandline interface:
```
boostmut_run -i input_directory_500ps -t 500
```
After the calculations have finished, the output can be processed with one of the tools in `boostmut_process` if needed. If the calculation of the mutations has been split up into separate parallel runs, the output has to be combined and rescaled. 
Combining can be done using `boostmut_process combine`. Rescaling the newly combined output file, or adding additional metrics can be done using `boostmut_process scale`. To obtain an easy human-readable excel version of the data, use `boostmut_process excel`. 


## Installation


