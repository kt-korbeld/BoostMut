import os
import re
import warnings
import numpy as np
import pandas as pd
import freesasa

import MDAnalysis
from MDAnalysis.analysis.base import AnalysisFromFunction
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",  module="Bio")
    from MDAnalysis.analysis import align
from MDAnalysis.analysis import rms

from scipy.optimize import curve_fit
from scipy.stats import poisson
import seaborn as sns
from sklearn.neighbors import KernelDensity
from scipy.stats import iqr

from .utils import load_universes
from .analysis import get_good_range_ix, get_sel_sasa, get_sasa_class

def generate_rmsf_data(universe_in, start=None,stop=None,step=1, ref_len=1):
    prot = universe_in.select_atoms('protein')
    class_dict = get_sasa_class(prot)
    rmsf_res, class_res = [], []
    firstres, lastres = min(prot.residues.resids), max(prot.residues.resids)
    average = align.AverageStructure(universe_in, universe_in, select='protein', ref_frame=0).run(start=start, stop=stop, step=step)
    average = average.results.universe
    counter = 2*ref_len+1

    for res in prot.residues:
        class_res.append(class_dict[res.resid])
        # skip glycine
        if res.resname == 'GLY':
            rmsf_res.append(0)
            continue
        # get range around res for local alignment, use ix instead of resid to deal with oligomers with same resid
        sel_res_range = prot.residues[np.isin(prot.residues.ix, get_good_range_ix(res, firstres, lastres, ref_len))]
        sel_res_range = 'id '+' '.join(sel_res_range.atoms.ids.astype(str))
        # align trajectory to average structure and get rmsf per sidechain
        if counter == 2*ref_len+1:
            aligner = align.AlignTraj(universe_in, average, select=sel_res_range, in_memory=True).run()
            counter = 0
        rmsf = rms.RMSF(res.atoms).run()
        rmsf = rmsf.results.rmsf
        rmsf_res.append(np.average(rmsf))
        counter+=1

    # save as df
    resnames = prot.residues.resnames
    resids = prot.residues.resids
    df_out = pd.DataFrame({'resid':resids, 'resname':resnames, 'rmsf':rmsf_res, 'class':class_res})
    return df_out

def generate_sasa_data(universe_in, start=None, stop=None, step=1):
    '''
    Perform an analysis of the hydrophobic solvent exposed surface
    an atom is defined as hydrophobic when it is either a carbon, or a hydrogen bound to a carbon.
    the sasa analysis is done on 3 selections:
    - the whole protein
    - 8A around the residue
    - just the residue
    It outputs the hydrophobic solvent exposed surface in A^2 for each selection.
    If the mutated residue is set to 'WT', it does it for each residue in the protein
    '''
    switch_AA = {'R':'ARG', 'H':'HIS', 'K':'LYS', 'D':'ASP', 'E':'GLU',
             'S':'SER', 'T':'THR', 'N':'ASN', 'Q':'GLN', 'C':'CYS',
             'G':'GLY', 'P':'PRO', 'A':'ALA', 'V':'VAL', 'I':'ILE',
             'L':'LEU', 'M':'MET', 'F':'PHE', 'Y':'TYR', 'W':'TRP'}
    # get sequence in single letter form
    rswitch_AA = {v: k for k, v in switch_AA.items()}
    hp_sel = '(element C or (element H and bonded element C))'
    prot = universe_in.select_atoms('protein')
    # get df with sasa for entire protein
    hp_cols = get_sel_sasa(prot, hp_sel).columns
    sasa_prot = AnalysisFromFunction(get_sel_sasa, universe_in.trajectory, prot, hp_sel).run(start=start, stop=stop, step=step)
    #print(sasa_prot.results.timeseries)
    df_sasa_prot = pd.DataFrame(data=np.average(sasa_prot.results.timeseries, axis=0), columns=hp_cols)
    # the larger analysis for the WT is done here
    resids = list(set(prot.residues.resids))
    resids.sort()
    sasa_res, sasa_val = [], []
    for i in resids:
        res_sel = 'resid {} and protein'.format(i)
        ind_r = prot.select_atoms('({}) and ({})'.format(res_sel, hp_sel)).atoms.indices
        resname = prot.select_atoms('({}) and ({})'.format(res_sel, hp_sel)).residues.resnames
        resname = list(set(resname))[0]
        #sasa_AA[rswitch_AA[resname]] = sasa_AA[rswitch_AA[resname]]+
        sasa_res.append(resname)
        sasa_val.append(np.sum(df_sasa_prot[ind_r].values))
    df_out = pd.DataFrame({'resname':sasa_res, 'sasa':sasa_val})
    return df_out

def process_benchmark_df_rmsf(df):
    # process data to plot as barplot
    plotting_df = pd.DataFrame({})
    amino_acids = list(set(df.resname))
    for aa in amino_acids:
        #plotting_df.join(df_B, how='outer')
        for cl in ['buried', 'partial', 'surface']:
            temp_df = pd.DataFrame(df[(df['resname']==aa) & (df['class'] == cl)]['rmsf'].to_numpy(), columns=[aa+'-'+cl])
            plotting_df = plotting_df.join(temp_df, how='outer')
    # put columns in a nice order
    surf_categories = ['buried','partial','surface']
    res_order = ['ALA','VAL','LEU','ILE','MET','PHE','TRP',
                     'CYS','PRO','GLY',
                     'SER','THR','ASN','GLN','TYR',
                     'ASP','GLU','HIS','LYS','ARG']
    desired_order = []
    for res in res_order:
        for c in surf_categories:
            desired_order.append(res+'-'+c)
    plotting_df = plotting_df.reindex(desired_order, axis=1)
    return plotting_df

def process_benchmark_df_sasa(df):
    switch_AA = {'R':'ARG', 'H':'HIS', 'K':'LYS', 'D':'ASP', 'E':'GLU',
                 'S':'SER', 'T':'THR', 'N':'ASN', 'Q':'GLN', 'C':'CYS',
                 'G':'GLY', 'P':'PRO', 'A':'ALA', 'V':'VAL', 'I':'ILE',
                 'L':'LEU', 'M':'MET', 'F':'PHE', 'Y':'TYR', 'W':'TRP'}
    # process data to plot as barplot
    plotting_df = pd.DataFrame({})
    amino_acids = ['R','H','K','D','E','S','T','N','Q','C','G','P','A','V','I','L','M','F','Y','W']
    for aa in amino_acids:
        temp_df = pd.DataFrame(df[(df['resname']==switch_AA[aa])]['sasa'].to_numpy(), columns=[aa])
        plotting_df = plotting_df.join(temp_df, how='outer')
    # put columns in a nice order
    plotting_df = plotting_df.reindex(amino_acids, axis=1)
    return plotting_df

def make_dens_step(dens):
    '''
    take any distribution of values and guarantee a pairwise monotonic decrease
    by setting all values lower but before a given y value to that y value.
    i.e it 'fills in' any local minima in the distribution to guarantee monotonic decrease
    '''
    # copy array and sort from highest to lowest
    dens_step = dens.copy()
    densorder = dens.copy()
    densorder.sort()
    highestval = 0
    # go from highest to lowest y value
    for i in densorder:
        # find highest x value where y value is found
        loc = max(np.where(dens == i))
        if len(loc) != 1:
            loc = max(loc)
        # take all x values lower than highest x value
        lower = np.where((dens < i))[0]
        if len(lower) == 0:
            continue
        # set lower x values to highest y value
        lower = lower[lower < loc]
        dens_step[lower] = i
    return dens_step

def make_benchmark_curves(plotting_df, x_array):
    y_array = np.array([])
    for col in plotting_df.columns:
        #print(col)
        data = plotting_df[plotting_df[col].notna()][col]
        if data.mean() == 0 or len(data.values) < 3:
            y_array = np.vstack((y_array, np.ones(500)))
            continue
        # fit a gaussian kde, set bandwith using standard trick
        bandwith = 1.06*min(data.std(), iqr(data)/1.34)*len(data)**(-1/5)
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwith).fit(data.to_numpy()[:, np.newaxis])
        dens = np.exp(kde.score_samples(x_array[:, np.newaxis]))
        dens = dens/(max(dens))
        if len(y_array) == 0:
            y_array = make_dens_step(dens)
            #y_array2 = dens
        else:
            y_array = np.vstack((y_array, make_dens_step(dens)))
            #y_array2 = np.vstack((y_array2, dens))
    return y_array

def make_benchmark_rmsf(input_dir, trajname='^[\w\d].*\.xtc$', topname='^[\w\d].*\.tpr$', outputname='benchmark_rmsf_out.csv',
                    start=None, stop=None, step=1):
    dfs_rmsf = []
    for prot in [i for i in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, i))]:
        print('processing: ', prot)
        # get all xtc files in directory and all pdb files with the same name
        universes = load_universes(os.path.join(input_dir, prot), topname, trajname)
        dfs_rmsf.extend([generate_rmsf_data(u, start=start,stop=stop,step=step) for u in universes])
    # concat and process all dfs
    df_rmsf = pd.concat(dfs_rmsf, ignore_index=True)
    df_rmsf = process_benchmark_df_rmsf(df_rmsf)
    # turn into benchmark curves
    x_array = np.linspace(0, 2, 500)
    y_array_rmsf = make_benchmark_curves(df_rmsf, x_array)

    bm_rmsf = pd.DataFrame(data=np.round(y_array_rmsf, 5), columns=np.round(x_array, 5), index=df_rmsf.columns)
    bm_rmsf.to_csv(outputname)

def make_benchmark_sasa(input_dir, trajname='^[\w\d].*\.xtc$', topname='^[\w\d].*\.tpr$', outputname='benchmark_sasa_out.csv',
                    start=None, stop=None, step=1):
    dfs_sasa = []
    for prot in [i for i in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, i))]:
        print('processing: ', prot)
        # get all xtc files in directory and all pdb files with the same name
        universes = load_universes(os.path.join(input_dir, prot), topname, trajname)
        dfs_sasa.extend([generate_sasa_data(u, start=start,stop=stop,step=step) for u in universes])
    # concat and process all dfs
    df_sasa = pd.concat(dfs_sasa, ignore_index=True)
    df_sasa = process_benchmark_df_sasa(df_sasa)
    # turn into benchmark curves
    x_array = np.linspace(0, 150, 500)
    y_array_sasa = make_benchmark_curves(df_sasa, x_array)
    bm_sasa = pd.DataFrame(data=np.round(y_array_sasa, 5), columns=np.round(x_array, 5), index=df_sasa.columns)
    bm_sasa.to_csv(outputname)
