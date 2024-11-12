# default python libraries
import re
import os
import itertools
from importlib import resources

# common data analysis packages
import pandas as pd
import numpy as np

# MDAnalysis and specific functions needed from MDAnalysis
from MDAnalysis import Universe
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis
from MDAnalysis.analysis import distances

#from MDAnalysis.analysis import rms, align
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from MDAnalysis.analysis import rms, align

from MDAnalysis.analysis.base import AnalysisFromFunction

# packages for secondary structure and solvent accesible surface
import pydssp
import freesasa

# yasara hydrogen bond energy function
# =======================================================================================================
def ehbo_yasara(universe_in, hbonds,
                E_cutoff=6.25, d_cutoff=2.6, maxhbonds=2, scale_bifurcation=True, padded=True):
    '''
    Takes in a universe and a list of hydrogen bonds obtained through HydrogenBondAnalysis, and
    returns a list of energies as calculated by formula described in the ListHBo command in YASARA:

         EnergyHBo = 25 * (2.6-max(Dist_h_a, 2.1) / 0.5) * Scale_d_h_a * Scale_h_a_x

    Where Dist_h_a is the distance between Hydrogen-Acceptor, and
    Scale_d_h_a depends on the angle formed by Donor-Hydrogen-Acceptor:

                       0   in range 0-100
        Scale_d_h_a =  0-1 in range 100-165
                       1   in range 165-180

    Scale_h_a_x depends on the angle formed by Hydrogen-Acceptor-X, where the latter X is
    the atom covalently bound to the acceptor. the scale depends on whether X is a hydrogen or not:

                       0   in range 0-75
        Scale_h_a_h =  0-1 in range 75-85
                       1   in range 85-180

                       0   in range 0-85
        Scale_h_a_x =  0-1 in range 85-95
                       1   in range 95-180

    If the Acceptor forms more than one covalent bond, the lowest scaling factor is taken.
    The Scale_h_a_x accounts for bumps between the hydrogen and the atoms covalently bound to the acceptor.
    An additional Scale_h_d is used as a sanity check to make sure no nonsense hydrogen bonds get passed
    If scale_bifurcation is set to True, the energies of bifurcated hydrogen bonds are scaled down.
    This is not part of the original Yasara function, but is more representative of the final interactions
    '''
    atomgroup = universe_in.select_atoms('all')
    # get information on the right frame from the hbond analysis results
    frame = int(atomgroup.ts.frame)
    hbonds_ts = hbonds[np.where(hbonds[:,0] == frame)]

    # get selections and measurements from hbond analysis
    index_out = hbonds_ts[:,1:4].astype(int)
    don = universe_in.atoms[hbonds_ts[:,1].astype(int)]
    hyd = universe_in.atoms[hbonds_ts[:,2].astype(int)]
    accept = universe_in.atoms[hbonds_ts[:,3].astype(int)]
    dist_h_a = distances.dist(hyd, accept, offset=0)[-1]
    dist_h_d = distances.dist(hyd, don, offset=0)[-1]
    angle_d_h_a = hbonds_ts[:,5]
    scale_h_d = np.array(dist_h_d <= 3).astype(int)
    scale_d_h_a = np.maximum(0, np.minimum(1, 1/65*(angle_d_h_a-100)))

    # find hydogen-acceptor-X pair for each atom x bonded to acceptor
    h_a_x = []
    [[h_a_x.append(h+a+x) for x in a.bonded_atoms if x in atomgroup] for h, a in zip(hyd, accept)]
    # get angle for each X and whether or not X is a hydrogen
    angle_h_a_x = [hax.angle.value() for hax in h_a_x]
    hyd_h_a_x = [x.type == 'H' for h, a, x in h_a_x]
    # calculate scale for H-A-X depending on whether it is a hydrogen or not
    scale_h_a_x = [max(0, min(1, 1/10*(angle-75))) if hyd else
                   max(0, min(1, 1/10*(angle-85))) for angle, hyd in zip(angle_h_a_x, hyd_h_a_x)]
    scale_h_a_x = np.array(scale_h_a_x)

    # match entry with lowest h_a_x scale for each unique Hydrogen-Acceptor
    hax_ind = np.array([hax.indices for hax in h_a_x])
    scale_h_a_x_matched = []
    for h, a in zip(hyd, accept):
        ind = [np.where((hax_ind[:,0] == h.index) & (hax_ind[:,1] == a.index))]
        scale = min(scale_h_a_x[ind[0][0]])
        scale_h_a_x_matched.append(scale)
    scale_h_a_x = scale_h_a_x_matched

    # calulate energy, set to 0 all results where bad angles/distances cause negative energy
    ehbo_yasara = 25*(d_cutoff-np.maximum(dist_h_a, 2.1))/(d_cutoff-2.1) * scale_d_h_a * scale_h_a_x * scale_h_d
    ehbo_yasara[ehbo_yasara <= E_cutoff] = 0
    ehbo_yasara = np.round(ehbo_yasara, 2)

    # Go over each acceptor and only keep the top (maxhbonds) hbonds
    index_accept = hbonds_ts[:,3].astype(int)
    unique_a, counts_a = np.unique(index_accept, return_counts=True)
    for ua, ca in zip(unique_a, counts_a):
        if ca > 1:
            ind = np.where(index_accept==ua)
            e = ehbo_yasara[ind]
            ind = ind[0][np.argsort(e)[::-1][maxhbonds:]]
            ehbo_yasara[ind] = 0

    # go over each hydrogen and only keep top (maxhbonds) hbonds
    index_h = hbonds_ts[:,2].astype(int)
    unique_h, counts_h = np.unique(index_h, return_counts=True)
    for uh, ch in zip(unique_h, counts_h):
        if ch > 1:
            ind = np.where(index_h==uh)
            e = ehbo_yasara[ind]
            ind = ind[0][np.argsort(e)[::-1][maxhbonds:]]
            ehbo_yasara[ind] = 0

    # scale down bifurcated bonds based on 10.1073/pnas.1319827111
    if scale_bifurcation:
        for ua, ca in zip(unique_a, counts_a):
            if ca > 1:
                ind = np.where(index_accept==ua)
                e = ehbo_yasara[ind]
                if len(e[e==0]) == len(e)-1:
                    continue
                ind = ind[0][np.where(e!=np.max(e))]
                ehbo_yasara[ind] = 15/25 * ehbo_yasara[ind]
    if padded:
        # pad the output array to a consistent size of max possible hbonds so no ragged lists are formed
        padded_len = int(len(universe_in.atoms)/2 - len(ehbo_yasara))
        return np.pad(ehbo_yasara, (0,padded_len), 'constant', constant_values=np.nan)
    else:
        return ehbo_yasara

def get_energy(universe_in, hbonds_real, e_hbonds_real, selection):
    '''
    given a universe, a set of hydrogen bonds, its correspoding energy, and a selection,
    give the energy of all protein-protein hydrogen bonds within that selection. 
    '''
    frames = [i.frame for i in universe_in.trajectory]
    # get indices for selection and protein
    selind = universe_in.select_atoms(selection).indices
    protind = universe_in.select_atoms('protein').indices
    anysel = np.any(np.isin(hbonds_real[:,1:4].astype(int), selind), axis=1)
    allprot = np.all(np.isin(hbonds_real[:,1:4].astype(int), protind), axis=1)
    # get energy for all bonds between protein
    e_hbond_sel = []
    for frame in frames:
        currentframe = (hbonds_real[:,0] == frame)
        e_hbond_sel.append(sum(e_hbonds_real[currentframe & anysel & allprot]))
    return e_hbond_sel

def get_wbridge(universe_in, hbonds_real, e_hbonds_real, selection, wbridge_penalty=(4.184*7.7)):
    '''
    given a universe, a set of hydrogen bonds, its correspoding energy, and a selection,
    give the energy of all waterbridges within that selection.
    The default penalty for fixing a water out of solution is 7.7 kcal/mol.
    taken from: 10.1110/ps.8.10.1982
    '''
    frames = [i.frame for i in universe_in.trajectory]
    # get indices for selection, protein and water
    selind = universe_in.select_atoms(selection).indices
    hohind = universe_in.select_atoms('resname HOH or resname SOL or resname TIP3').indices
    protind = universe_in.select_atoms('protein').indices
    anysel = np.any(np.isin(hbonds_real[:,1:4].astype(int), selind), axis=1)
    anywater = np.any(np.isin(hbonds_real[:,1:4].astype(int), hohind), axis=1)
    anyprot = np.any(np.isin(hbonds_real[:,1:4].astype(int), protind), axis=1)
    # get all hbonds with both water and protein
    e_wbridge = []
    for frame in frames:
        currentframe = (hbonds_real[:,0] == frame)
        # get waters bound to selection
        hbonds_sel_water = hbonds_real[currentframe & anysel & anywater]
        waterselind = hbonds_sel_water[:,1:4][np.isin(hbonds_sel_water[:,1:4], hohind)].astype(int)
        anywatersel = np.any(np.isin(hbonds_real[:,1:4].astype(int), waterselind), axis=1)
        # get all hbonds between protein and water containing previously selected waters
        hbonds_wbridge = hbonds_real[currentframe & anywatersel & anyprot]
        water_wbridge = hbonds_wbridge[:,1:4][np.isin(hbonds_wbridge[:,1:4], hohind)].astype(int)
        # get all (O of ) waters that occur in more than 1 bond since these contribute to a water bridge
        unique, count = np.unique(water_wbridge, return_counts=True)
        wbridge_waters = unique[count > 1]
        in_waterbridge = np.any(np.isin(hbonds_real[:,1:4].astype(int), wbridge_waters), axis=1)
        # for each water, get selection of hbonds containing that water
        hbonds_wbridge = hbonds_real[currentframe & in_waterbridge][:,1:4].astype(int)
        bondsperw = [hbonds_wbridge[np.any(np.isin(hbonds_wbridge, water), axis=1)] for water in wbridge_waters]
        # if nr of residues per water is <= 2, it is a bridga between one residue and itself, exclude bonds
        wbridge_self = np.array([len(universe_in.atoms[i.flatten()].residues) <= 2 for i in bondsperw])
        if len(wbridge_self) != 0:
            wbridge_waters = wbridge_waters[~wbridge_self]
        in_waterbridge = np.any(np.isin(hbonds_real[:,1:4].astype(int), wbridge_waters), axis=1)
        # add up the energies of all contributing hbonds and substract entropic penalty for water
        e_wbridge.append(sum(e_hbonds_real[currentframe & in_waterbridge]) - len(wbridge_waters)*wbridge_penalty)
    # if the water bridges are poor enough to to be negative, ignore and set to 0
    e_wbridge = np.array(e_wbridge)
    e_wbridge[e_wbridge < 0] = 0
    return e_wbridge

def get_unsat(universe_in, hbonds_real, selection, unsat_penalty=1, return_all=False):
    '''
    given a universe, a set of hydrogen bonds as outputted by the MDAnalysis hydrogen bond package,
    and a selection, return the number of potential hydrogen bond partners that are not satisfied
    within that selection. i.e. the hydrogen bond parters that are within the selection but not
    in the set of hydrogen bonds.
    unsat_penalty allows one to set the weight of each unsaturated hydrogen bond
    return_all can be used t oreturn all hydrogen bond partners, not just the unsatisfied ones
    '''
    frames = [i.frame for i in universe_in.trajectory]
    # all atoms that can form hbonds within selection
    hbond_sel = ('protein and ((type H and bonded ((type O or type N))) or'
                '(type O or (type N and not bonded type H)))')
    # narrow down selection to non-carbons in protein before doing expensive bond checking
    prot_nc = universe_in.select_atoms('protein and not type C')
    hbond_atm = prot_nc.select_atoms('({}) and ({})'.format(hbond_sel, selection)).indices
    hbonds_ha = hbonds_real[:,[2,3]].astype(int)
    if return_all:
        return hbond_atm
    # get number of unsaturated bonds
    e_unsat = []
    for frame in frames:
        currentframe = (hbonds_real[:,0] == frame)
        # for current frame, check which atoms that cannot form
        unsat = hbond_atm[~np.isin(hbond_atm, hbonds_ha[currentframe])]
        e_unsat.append(unsat_penalty * len(unsat))
    return e_unsat

# all functions dealing with secondary structure and helix caps
# =======================================================================================================

def get_pydssp(atomgroup):
    '''
    gives the secondary structure for any backbone in a given atom group
    using the pydssp implementation of dssp
    '''
    bb_atoms = ['N', 'CA', 'C', 'O']
    backbone_res = atomgroup.select_atoms('backbone').residues
    coord = np.array([[atom.position for atom in r.atoms if atom.name in bb_atoms] for r in backbone_res])
    secstr = pydssp.assign(coord, out_type='c3')
    return secstr

def matchmotifs(sequence, cap, startorend,  buffer=5):
    '''
    takes a list containing the sequence of residues in single letter notation,
    and residue number where an alpha-helix ends
    returns whether any of the motifs are present around the cap
    based on doi.org/10.1002/pro.5560070103
    '''
    # define the capping motifs for either N- or C-terminus of a-helix
    N_capping_motifs = [((-1, 5), 'hxpxhx'), ((-1, 5), 'hxpxph'), ((-2, 5), 'hpxpxhx'),
                        ((-2, 5), 'hpxpxph'), ((-3, 5), 'hppxpxhx'), ((-3, 5), 'hppxpxph')]
    C_capping_motifs = [((-4, 2), 'HxpxGh'), ((-4, 2), 'lxpxGh'), ((-4, 3), 'Hxpxnxh'),
                        ((-4, 4), 'Hxpxnxph'), ((-4, 4), 'HxxxGpxh'), ((-4, 5), 'HxxxGpxph'),
                        ((-4, 4), 'HxxxPpxh'), ((-4, 5), 'HxxxPpxph')]
    # define res classes used in the motifs: p:polar, n:non-b-branched, x:indifferent, h:hydrophobic,
    #H: hydrophobic inside helix (alkyl-side chains of K or R can be used)
    #l: polar residue long enough to make hydrophobic contact in the C3/C'G motif
    resclasses = {'h':'[AVILMFWCH]','H':'[AVILMFWCHKR]','l':'[KRFYWM]','p':'[GSTNQDEKRH]',
                  'n':'[ALMFWCGSNQYDHEKR]','x':'[AVILMFPWCGSTNQYDHEKR]','G':'G','P':'P'}
    # check if start or end is specified
    if startorend == 'start':
        motifs = N_capping_motifs
    elif startorend == 'end':
        motifs = C_capping_motifs
    else:
        print('not start or end')
        return
    # go over each cap and check if motif is present
    motifmatch = []
    for pos, motif in motifs:
        # add buffer around range in case the secondary structure is off
        posind = tuple(map(sum, zip(pos, (-buffer, buffer))))
        seq = ''.join(sequence[posind[0]+cap:posind[1]+cap]) 
        #convert capping motifs into regex using dictionary of resclasses
        regexmotif =''.join([resclasses[i] for i in motif])
        foundmotif = not (None == re.search(regexmotif, seq))
        motifmatch.append(foundmotif)
        #print out the found cap and its matching motif
    return any(motifmatch)

def get_helixcaps(atomgroup, starts=[], ends=[], output='index'):
    '''
    given a universe or atomgroup, takes the protein sequence, assigns dssp,
    finds the starts and ends of each helix, and checks if a capping motif is broken.
    The checking of the capping motifs is done using the matchmotifs() function
    '''
    # get sequence and secondary structure
    prot = atomgroup.select_atoms('protein')
    sequence = list(prot.residues.sequence().seq)
    secstr = get_pydssp(prot)
    if len(starts) == 0 or len(ends) == 0:
        # find where stretches of H (helix) start/end
        ishelix = np.hstack([ [False], (secstr == 'H'), [False] ]).astype(int)
        starts_ends = np.diff(ishelix.astype(int))
        starts, ends = np.where(starts_ends == 1)[0], np.where(starts_ends == -1)[0]
        # exclude helices shorter than 4 residues (minimum required for 1 a-helix turn)
        starts = [starts for starts, ends in zip(starts, ends) if abs(starts-ends) >= 4]
        ends = [ends for starts, ends in zip(starts, ends) if abs(starts-ends) >= 4]
    # give boolean of whether capping motifs are present at these starts/ends
    startmotifs = np.array([matchmotifs(sequence, start, 'start') for start in starts])
    endmotifs = np.array([matchmotifs(sequence, end, 'end') for end in ends])
    if output=='index':
        return np.array(starts)[startmotifs], np.array(ends)[endmotifs]
    if output=='count':
        startcount = len(startmotifs[startmotifs == True])
        endcount = len(endmotifs[endmotifs == True])
        return startcount+endcount
    if output=='ratio':
        startratio = len(startmotifs[startmotifs == True])/len(startmotifs)
        endratio = len(endmotifs[endmotifs == True])/len(endmotifs)
        return startratio, endratio

def get_helix_score(resnames, secstr, buffer=5):
    '''
    if a list of residue names and secondary structure assignments is given,
    returns a helicity score based on the helix propensities of each residue in an a-helix. 
    the propensities are taken from 10.1016/s0006-3495(98)77529-0 
    assuming a pH of 7 (+ charged ARG, LYS, HIS, - charged ASP, GLU )
    the buffer are the number of residues near the start and end of the helix that are ignored.
    this prevents problems with penalizing beneficial residues in helix motifs 
    '''
    helix_propensities = {'ALA':0, 'LEU':0.16, 'MET':0.24, 'ARG':0.21, 'LYS':0.26, 
                          'GLN':0.39, 'GLU':0.4, 'ILE':0.41, 'TRP':0.49, 'SER':0.5, 
                          'TYR':0.53, 'PHE':0.54, 'VAL':0.61, 'THR':0.66, 'ASN':0.65, 
                          'HIS':0.66, 'CYS':0.68, 'ASP':0.69, 'GLY':1, 'PRO':3.16}
    helix_score = 0
    for ind in range(0, len(secstr)):
        ss = secstr[ind-(buffer):ind+(buffer+1)]
        res = resnames[ind]
        if np.all(ss == 'H'):
            helix_score+=helix_propensities[res]
    return helix_score

# all functions dealing with solvent accesible surface
# =======================================================================================================

def get_sasa(atomgroup, selection='R'):
    '''
    for a given MDAnalysis universe or atom group,
    give the solvent accesible surface.
    selection can return sasa for A (atom), R (residue) or S (structure)
    '''
    #set algorithm to shrake-rupley, set probe radius if neccesary
    param = freesasa.Parameters()
    param.setAlgorithm('ShrakeRupley')
    param.setProbeRadius(1.4)
    # standard radii taken from the biopython implementation of shake-rupley
    radii_atomtypes= {"H": 1.200, "HE": 1.400, "C": 1.700, "N": 1.550, "O": 1.520, "F": 1.470,
                  "NA": 2.270, "MG": 1.730, "P": 1.800, "S": 1.800, "CL": 1.750, "K": 2.750,
                  "CA": 2.310, "NI": 1.630, "CU": 1.400, "ZN": 1.390, "SE": 1.900, "BR": 1.850,
                  "CD": 1.580, "I": 1.980, "HG": 1.550}
    # calcuating sasa requires coordinates and radii per atom
    atomtypes = atomgroup.atoms.types
    radii = np.vectorize(radii_atomtypes.get)(atomtypes)
    coords = atomgroup.atoms.positions.flatten()
    sasa_result = freesasa.calcCoord(coords, radii, param)
    sasa_atom = np.array([sasa_result.atomArea(i) for i, a in enumerate(atomgroup.atoms)])
    # sasa per atom
    if selection == 'A':
        return sasa_atom
    # sasa per residue
    if selection == 'R':
        # if array is not equal to array shifted by one, then a transition took place
        resids = atomgroup.atoms.resids
        resids_shifted = np.concatenate((resids[-1:], resids[0:-1]))
        transitions = np.concatenate((np.where(resids != resids_shifted)[0], [len(resids)]))
        # use transitions to get start/end of atoms with same resid to split up by residues
        return np.array([np.sum(sasa_atom[start:end]) for start, end in zip(transitions[0:-1], transitions[1:])])
    # sasa for entire structure
    if selection == 'S':
        return np.sum(sasa_atom)

def to_sasa_class(value, buried_cutoff=0, surface_cutoff=0.2):
    if value == buried_cutoff:
        return 'buried'
    if buried_cutoff < value <= surface_cutoff:
        return 'partial'
    if value > surface_cutoff:
        return 'surface'

def get_sasa_class(atomgroup, return_resname=False):
    '''
    Given a universe or atomgroup, retuns the sasa class for each residue,
    consisting of 'buried', 'partial' and 'surface'.
    '''
    #MD based estimations of average SASA for each of the amino acids alone in space, from 10.1007/s00894-009-0454-9
    total_sasa_res = {'ALA':209.02, 'ARG':335.73, 'ASN':259.85, 'ASP':257.99, 'CYS':240.5,
                      'GLN':286.76, 'GLU':285.03, 'GLY':185.15, 'HIS':290.04, 'ILE':273.46,
                      'LEU':278.44, 'LYS':303.43, 'MET':291.52, 'PHE':311.30, 'PRO':235.41,
                      'SER':223.04, 'THR':243.55, 'TRP':350.68, 'TYR':328.82, 'VAL':250.09}
    prot = atomgroup.select_atoms('protein')
    resids = prot.residues.resids
    resnames = prot.residues.resnames
    sasa = get_sasa(prot, selection='R')
    sasa_classes = [to_sasa_class(s/total_sasa_res[r]) for r, s in zip(resnames, sasa)]
    sasa_class_dict = dict([*zip(resids, sasa_classes)])
    if return_resname:
        resname_dict = dict([*zip(resids, resnames)])
        return resname_dict, sasa_class_dict
    else:
        return sasa_class_dict

def get_hydrocarb_sasa(universe_in):
    '''
    gets the total sasa for all the hydrocarbon atoms (i.e carbons or hydrogens attached to carbons)
    '''
    sasa_atom = get_sasa(universe_in, selection='A')
    hydrocarb = universe_in.select_atoms('type C or type H and bonded type C')
    hydrocarb_sasa = sum([sasa for sasa, atom in zip(sasa_atom, universe_in.atoms) if atom in hydrocarb])
    return hydrocarb_sasa

def get_sel_sasa(universe_in, selection, sum_out=False):
    '''
    gets the total sasa for all the selected atoms
    '''
    sasa_atom = get_sasa(universe_in, selection='A')
    sel_atoms = universe_in.select_atoms(selection)
    sel_sasa = [sasa for sasa, atom in zip(sasa_atom, universe_in.atoms) if atom in sel_atoms]
    if sum_out:
        sel_sasa = sum(sel_sasa)
        return sel_sasa
    else:
        sel_sasa = np.array(sel_sasa).reshape(1,len(sel_sasa))
        sel_sasa = pd.DataFrame(data = sel_sasa, columns=sel_atoms.atoms.indices)
    return sel_sasa

# miscelaneous functions
# =======================================================================================================

def get_surround_sel(universe_in, selection, dist_cutoff=8, byres='byres'):
    '''
    the 'around' keyword in MDAnalysis is sometimes unreliable
    this function works as a select_atoms('byres around X selection') operation,
    but also doublechecks the distances and excludes wrongly selected atoms
    '''
    surrounding = universe_in.select_atoms('{} around {} {}'.format(byres, dist_cutoff, selection))
    atoms = universe_in.select_atoms('{}'.format(selection))
    real_sel = len(surrounding.atoms.positions)*[False]
    for atom in atoms.atoms.positions:
        atom_rep = np.vstack(len(surrounding.atoms.positions)*[atom])
        dist = np.linalg.norm(atom_rep-surrounding.atoms.positions, axis=1)
        real_sel = (dist < dist_cutoff+4) | real_sel
    return surrounding[real_sel]

def dens_to_score(test, x_val, y_dens):
    '''
    this function uses a given distribution of rmsf and sasa
    and its corresponding x values obtained from the simulations of 77 small proteins.
    the distributions have been smoothened and the max height set to 1,
    providing a way to score the RMSF/SASA of each residue from 1 to 0,
    where 1 is realistic and 0 is unrealistic depending on the distribution
    '''
    #make sure input is a np array
    if type(test) != np.ndarray:
        if type(test) != list:
            test = [test]
        test = np.array(test)
    # find first highest point in distribution, everything before that is also 1
    x_ymax = min(x_val[np.where(y_dens == max(y_dens))])
    test[test<x_ymax] = x_ymax
    output = np.interp(test, x_val, y_dens)
    # if the test value is higher than the given range, set to max of range
    output[np.where(test>max(x_val))] = 0
    return output

def get_good_range_ix(res, firstres, lastres, ref_len):
    '''
    get a selection of ref_len residues around a given residue,
    making sure that sufficient residues are taken if near the start/end
    '''
    #print(res.resid, res.ix, res.resid - ref_len < firstres)
    # if there are insufficient residues at the front, use res at the end
    if res.resid - ref_len < firstres:
        ix_res_range = [res.ix, res.ix+2*ref_len]
    # if there are insufficient residues at the end, use res at the front
    elif res.resid + ref_len > lastres:
        ix_res_range = [res.ix-2*ref_len, res.ix]
    # otherwise just take normal range
    else:
        ix_res_range = [res.ix-ref_len, res.ix+ref_len]
    return [*range(*ix_res_range)]

def load_benchmark_data(filename, location='BoostMut.benchmarks'):
    '''
    for a given location, load the benchmark data in a df and return the df
    '''
    with resources.open_text(location, filename) as file:
        df = pd.read_csv(file, index_col=0)
    return df

def find_xy_dens(dens_df, classification):
    '''
    given a classification ('ARG-buried'), finds the curve of expected values
    in the input dens_df dataframe. the dataframe should have the
    classification names as indices of the rows.
    '''
    x = dens_df.columns.to_numpy().astype(float)
    y = dens_df.loc[classification].to_numpy().astype(float)
    return x, y

def get_saltbridge(universe_in, selection, distance=4):
    '''
    for a given universe, finds number of saltbridges present
    '''
    sel_sb_pos = '(resname ARG or resname LYS and type N and not backbone)'
    sel_sb_neg = '(resname ASP or resname GLU and type O and not backbone)'
    # find all positive res engaged in a saltbridge and all negative res engaged in a salt bridge
    sb_pos = universe_in.select_atoms('({0} and around {3} {1}) and {2}'.format(sel_sb_pos, sel_sb_neg, selection, distance))
    sb_neg = universe_in.select_atoms('({1} and around {3} {0}) and {2}'.format(sel_sb_pos, sel_sb_neg, selection, distance))
    # since one Arginine might have multiple partners, use max between positive and negative as true count
    return max(len(sb_pos), len(sb_neg))

def get_disulfide(universe_in):
    '''
    counts number of disulfide bonds in a given
    '''
    disulfidebond = 0
    for bond in universe_in.select_atoms('type S').bonds:
        residues = bond.atoms.residues.resnames
        if np.all(residues == 'CYS') and len(residues) == 2:
            disulfidebond+=1
    return disulfidebond

def iterate_analysis(mut_universes, function, mut_ids=[]):
    '''
    iterate analysis of a function over multiple universes,
    and average the result
    '''
    out = []
    for mut_universe in mut_universes:
        vars_out = function(mut_universe.copy(), mut_ids=mut_ids)
        if len(out) == 0:
            out = vars_out
        out = [np.round(np.average(np.array([i,j]), axis=0), 4) for i, j in zip(out, vars_out)]
    return tuple(out)

def write_sub_trajectory(universe, filename_out='out.xtc', start=0, end=0):
    '''
    given a trajectory, save a new one from start to end
    '''
    with MDAnalysis.Writer(filename_out) as W:
        for ts in u.trajectory[start:end]:
            W.write(u)

def reject_traj(universes_in):
    '''
    given a list of multiple MDAnalysis universes,
    removes the one that has the highest RMSD with all the other universes
    '''
    if len(universes_in) == 1:
        return universes_in
    universes_out = universes_in.copy()
    indices = [i for i in itertools.combinations([*range(len(universes_in))],2)]
    # find pairwise rmsd between each of the universes
    rmsd_between_universe = []
    for universe_1, universe_2 in itertools.combinations(universes_in,2):
        universe_1, universe_2 = universe_1.copy(), universe_2.copy()
        rmsd = rms.RMSD(universe_1, universe_2, select='backbone', ref_frame=0)
        rmsd.run()
        # take last column of results containing rmsd and average over trajectory
        rmsd = np.average(rmsd.results.rmsd[:,-1])
        rmsd_between_universe.append(rmsd)
    # for each uinverse find average rmsd of all pairwise rmsds containing that universe
    avg_rmsd_universe = []
    for i in range(len(universes_in)):
        rmsd_i = [rmsd for rmsd, ind in zip(rmsd_between_universe, indices) if i in ind]
        avg_rmsd_universe.append(np.average(rmsd_i))
    worst_ind = np.where(np.array(avg_rmsd_universe) == max(avg_rmsd_universe))[0][0]
    universes_out.pop(worst_ind)
    return universes_out















