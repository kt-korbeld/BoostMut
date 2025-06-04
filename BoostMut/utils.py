# default python libraries
import re
import os
import itertools
from importlib import resources

# other packages
import pandas as pd
import numpy as np
from MDAnalysis import Universe


def bondstab_to_bonds(filename):
    '''
    in order to add bonds to a system,
    they have to be read in as a list of tuples
    '''
    # read in file
    with open(filename, 'r') as tab:
        bonds_raw = tab.readlines()
    # parse bonds in a manner MDAnalysis can accept
    bonds = []
    for bondline in bonds_raw:
        bond = bondline[:-1].split(' ')
        # yasara is 1-index, python is 0-index, so -1
        bond = tuple([int(i)-1 for i in bond])
        bonds.append(bond)
    return bonds

def load_universes(dir_path, topname, trajname, bondstabname='', guess_bonds=False):
    '''
    load all the universes in a given directory.
    for each universe, topology and trajectories are identified based on a given regex
    accepts all the topology and trajectory formats compatible with MDAnalysis
    if bondinfo is missing, from the topology files, guess_bonds can be set to True,
    or a seperate file containing bond info can be used to load in the bonds
    '''
    universes = []
    # find all files in directory that match topology regex
    top_files = list(filter(re.compile(topname).match, os.listdir(dir_path)))
    top_files = [os.path.join(dir_path, top) for top in top_files]
    # find all files in directory that match trajectory regex
    traj_files = list(filter(re.compile(trajname).match, os.listdir(dir_path)))
    traj_files = [os.path.join(dir_path, trj) for trj in traj_files]
    # check if neccesary files are present
    if len(traj_files) == 0 or len(top_files) == 0:
        print('trajectory or topology file not found')
        return universes
    # in case mutliple topology files are found try each and use first one that works
    for top in top_files:
        try:
            if guess_bonds:
                universes = [Universe(top, traj, guess_bonds=True) for traj in traj_files]
            else:
                universes = [Universe(top, traj) for traj in traj_files]
            break
        except:
            print(top, 'did not provide the correct topology for:\n', traj_files)
            universes = []
            continue
    if len(bondstabname) == 0:
        return universes
    # if seperate tab file with bond info is provided, add bonds from that
    bond_files = list(filter(re.compile(bondstabname).match, os.listdir(dir_path)))
    bond_files = [os.path.join(dir_path, bf) for bf in bond_files]
    # in case mutliple bond files are found try each and use first one that works
    for bf in bond_files:
        try:
            bonds = bondstab_to_bonds(bf)
            [u.add_TopologyAttr('bonds', bonds) for u in universes]
        except:
             print(bf, 'did not provide the correct bond info for:\n', traj_files)
             universes = []
    return universes

def scale_df(df_in, scale_custom={}, returnmap=False):
    '''
    Takes in a dataframe with the final results and scales each value according to stdev,
    and multiplies with 1 or -1 depending on wether an increase is good or bad.
    this results in an output of values near 1 for which higher values are always desirable.
    '''
    betterorworsemap = {'e_hbonds_p':1,'e_hbonds_s':1, 'e_hbonds_r':1,
                        'hbonds_unsat_p':-1, 'hbonds_unsat_s':-1, 'hbonds_unsat_r':-1,
                        'rmsf_bb_p':-1, 'rmsf_bb_s':-1, 'rmsf_bb_r':-1,
                        'score_sc_p':1, 'score_sc_s':1, 'score_sc_r':1,
                        'hpsasa_p':1, 'hpsasa_s':1, 'hpsasa_r':1,
                        'saltb_p':1, 'saltb_s':1, 'saltb_r':1,
                        'capcount':1, 'helicity':-1, 'disulfide':1}
    scale_complete = betterorworsemap|scale_custom
    if returnmap:
        return scale_complete
    #flip sign depending on whether an increase in the metric is good or bad
    std_df = df_in.std()
    std_df[std_df == 0] = 0.1
    df_scaled = df_in/std_df
    for col in df_scaled.columns:
       	df_scaled[col] = df_scaled[col]*scale_complete[col]
    df_scaled = df_scaled.round(4)
    return df_scaled

def get_custom_scale(df_in, names, scales):
    '''
    given a set of names and scales, match the two into a custom scale map
    if only partial data is provided, use defaults to make reasonable assumptions
    '''
    standard_cols = scale_df(df_in, returnmap=True)
    nonstandard_cols = [i for i in df_in.columns if not i in standard_cols]
    print('found {} non-standard columns: '.format(len(nonstandard_cols)), nonstandard_cols)
    # if no nonstandard cols are present, ignore names and scales
    if len(nonstandard_cols) == 0:
        return {}
    # if no col names and a single scale is provided, scale all nonstandard cols by 1
    elif len(names) == 0 and len(scales) == 1:
        print('using scaling factor: {} for all columns: '.format(scales), nonstandard_cols)
        custom_scale = {i:j for i,j in zip(nonstandard_cols, len(nonstandard_cols)*scales)}
    # if no col names and provided scales matches the number nonstandard cols, use provided scale
    elif len(names) == 0  and len(scales) == len(nonstandard_cols):
        print('matching scaling factors {} to columns: {}'.format(scales, nonstandard_cols))
        custom_scale = {i:j for i,j in zip(nonstandard_cols, scales)}
    # if col names are provided but a mismatch exists, raise error
    elif len([i for i in nonstandard_cols if not i in names]) > 0:
        raise ValueError("specified column names do not match all nonstandard columns in input:", nonstandard_cols)
    elif len(scales) != len(names):
        raise ValueError("specified column names do not match number of scaling factors", names, scales)
    else:
        print('matching scaling factors {} to columns: {}'.format(scales, names))
        custom_scale = {i:j for i,j in zip(names, scales)}
    return custom_scale

def get_mutinfo(input_dir, mut_regex, mutfile, exclude=[]):
    '''
    based on an input directory and a regex pattern,
    for each subdirectory named after its mutation according to the regex pattern,
    return a the mutations, resids, and  subdirectory paths for each of the mutants
    if a mutfile is given, only return the outputs for the mutations in the mutfile
    '''
    # get all directories that match the given mutname regex
    mutant_dirs = list(filter(re.compile(mut_regex).match, os.listdir(input_dir)))
    mutant_dirs.sort(key=lambda x: int(re.findall(r'\d+',x)[-1]))
    mutsel = [re.search('[A-Z][0-9]+[A-Z]', mut).group() for mut in mutant_dirs]
    # if an input mutfile is given, only analyze mutations in the mutfile
    if len(mutfile) > 0:
        with open(mutfile, 'r') as tab:
            muts = [i.strip() for i in tab.readlines()]
        mutsel = [mut for mut in mutsel if mut in muts]
    # if mutations to exclude are given, remove excluded mutations from dirs and muts
    if len(exclude) > 0:
        mutsel = [i for i in mutsel if i not in exclude]
    # get the resnrs,  dirs and full path only for the selected mutations
    mutant_dirs = [mut for mut in mutant_dirs if any([i in mut for i in mutsel])]
    mutant_paths  = [os.path.join(input_dir, mut) for mut in mutant_dirs if os.path.isdir(os.path.join(input_dir, mut))]
    mutant_paths.sort(key=lambda x: int(re.findall(r'\d+',x)[-1]))
    resnrs = [int(re.findall('[0-9]+', i)[0]) for i in mutant_dirs]
    print('number of mutations: ', len(mutant_paths))
    return mutsel, resnrs, mutant_paths

def get_columnnames(analyses='hrsc', selection=['hrs:sr', 'c:p']):
    '''
    given an input specifying a set of analyses and selections,
    return the right set of columnnames in the right order for the dataframe
    '''
    analyses_map = {'h':['e_hbonds', 'hbonds_unsat'], 'r':['rmsf_bb', 'score_sc'],
                    's':['hpsasa'], 'c':['saltb', 'capcount','helicity','disulfide']}
    nosels = ['capcount','helicity','disulfide']
    # generate a dictionary mapping each analysis to a selection
    sel_dict = {}
    for entry in selection:
        entry = entry.strip().split(':')
        sel_dict = sel_dict | {i:entry[-1] for i in entry[0]}
    columns = []
    for an in 'hrsc':
        # if analysis is not specified, skip
        if not an in analyses:
            continue
        # if analysis has no specified selection, give all
        if an not in sel_dict.keys():
            sel_dict[an] = 'psr'
        # generate column names for given analysis and selection
        cols = analyses_map[an]
        for c in cols:
            if c in nosels:
                columns.extend([c])
            else:
                columns.extend([c+'_'+i for i in 'psr' if i in sel_dict[an]])
    return columns

def generate_excel(df_in, excel_out):
    '''
    given a dataframe with scaled output values, returns a formatted excel sheet
    '''
    df_out = df_in.copy()
    # calculate total score if not present
    if not 'total_score' in df_in.columns:
        df_out['total_score'] = df_in.sum(axis=1).values
    # put total score in first column and sort mutations by total score
    df_out = df_out[['total_score']+[i for i in df_out.columns if i != 'total_score']]
    df_out = df_out.sort_values(by='total_score', ascending=False)
    if excel_out[-5:] != '.xlsx':
         excel_out = excel_out+'.xlsx'
    writer = pd.ExcelWriter(excel_out, engine='xlsxwriter')
    df_out.to_excel(writer, sheet_name='Sheet1')
    workbook  = writer.book
    worksheet = writer.sheets['Sheet1']
    (max_row, max_col) = df_in.shape
    for i, col in enumerate(df_in.columns):
        if col != 'total_score':
            worksheet.conditional_format(1, 1, max_row, i+1,
                                         {'type': '3_color_scale',
                                          'minimum':-5, 'maximum':5,
                                          'min_color':'#E77270', 'mid_color':'#FFFFFF', 'max_color':'#6488C1'})
        else:
                worksheet.conditional_format(1, 1, max_row, i+1,
                                         {'type': '3_color_scale',
                                          'min_color':'#E77270', 'mid_color':'#FFFFFF', 'max_color':'#6488C1'})
    number_format = workbook.add_format({'num_format': '0.00'})
    worksheet.set_column(1,max_row,None,number_format)
    writer.close()


def combine_df(df_set=[]):
    df_complete = []
    for df in df_set:
        if len(df_complete) == 0:
            df_complete = df.copy()
            continue
        df_i = [i for i in df.index if i not in df_complete.index]
        for i in df_i:
            df_complete.loc[i] = df.loc[i]
    return df_complete

def add_predictors(df_in, predictors, pred_cols, pred_names):
    df_out = df_in.copy()
    # if only one name/column is specified but more files exist, extrapolate
    if pred_names == ['primary_pred'] and len(predictors) > 1:
        score_names = ['primary_pred_'+i for i in range(len(predictors))]
    else:
        score_names = pred_names
    if len(pred_cols) == 1 and len(predictors) > 1:
        score_cols = len(predictors)*pred_cols
    else:
        score_cols = pred_cols
    # load in each specified additional predictor
    for file, col, nam in zip(predictors, score_cols, score_names):
        df_score = pd.read_csv(file, index_col=0)
        df_out[nam] = df_score.loc[df_out.index].values
    return df_out


