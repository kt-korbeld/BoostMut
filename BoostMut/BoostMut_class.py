from .analysis import *
from importlib import resources

class BoostMut:
    def __init__(self, WT_universes, mut_ids=[], rmsf_loc='range_rmsf.csv', sasa_loc='range_sasa.csv', reject_trj=True):
        '''
        Since each metric is compared with the WT, and each metric is checked for different selections,
        the WT results for each metrics must be stored before a selection is made.
        the results for each residue are stored in a dataframe with one resid per column
        '''
        #reject worst trajectory from set if reject_trj is True
        if reject_trj:
            self.WT_universes = reject_traj(WT_universes)
        else:
            self.WT_universes = WT_universes
        # load benchmark data
        self.sasa_range = load_benchmark_data(sasa_loc)
        self.rmsf_range = load_benchmark_data(rmsf_loc)
        # define rigid set of resids for local surrounding of each residue
        surround_sel_ids = []
        example_prot = WT_universes[0].select_atoms('protein')
        resids = list(set(example_prot.residues.resids))
        resids.sort()
        for resid in resids:
            sel_s = 'resid {} and name CA'.format(resid)
            surround_sel = get_surround_sel(example_prot, sel_s, dist_cutoff=8)
            s_resids = surround_sel.residues.resids
            surround_sel_ids.append(s_resids)
        self.WTresids = resids
        self.WTsurround_sel = pd.DataFrame(data=surround_sel_ids, index=resids)
        # define empty WT variables to be replaced by do_analysis_WT
        self.WTe_hbonds = []
        self.WThbonds_unsat = []
        self.WTrmsf_bb = []
        self.WTscore_sc = []
        self.WThpsasa = []
        self.WTchecks = []
        self.WTsaltb = []

    def do_analysis_WT(self, mut_ids=[], analyses='hrsc'):
        '''
        Since each metric is compared with the WT, and each metric is checked for different selections,
        the WT results for each metrics must be stored before a selection is made.
        the results for each residue are stored in a dataframe with one resid per row
        the columns are usually the selection (p:protein, s:surrounding, r:residue),
        except for the checks, which are done over the whole protein.
        the checks have an entry per residue for consistency, but these are the same for each residue. 
        (c:helix caps, p:proline in helix, s:exposed sulphur, d:disulfide bridges)
        '''
        print('Analyzing WildType..')
        # do hbond analysis and average output over WT trajectories
        if 'h' in analyses:
            print('analyzing hydrogen bonds..')
            e_hbonds, hbonds_unsat = iterate_analysis(self.WT_universes, self.do_hbond_analysis, mut_ids)
            self.WTe_hbonds = pd.DataFrame(data=e_hbonds, index=mut_ids, columns=['p','s','r'])
            self.WThbonds_unsat = pd.DataFrame(data=hbonds_unsat, index=mut_ids, columns=['p','s','r'])
        # do RMSF analysis and average output over WT trajectories
        if 'r' in analyses:
            print('analyzing RMSF of backbone and sidechains..')
            rmsf_bb, score_sc = iterate_analysis(self.WT_universes, self.do_rmsf_analysis, mut_ids)
            self.WTrmsf_bb = pd.DataFrame(data=rmsf_bb, index=mut_ids, columns=['p','s','r'])
            self.WTscore_sc = pd.DataFrame(data=score_sc, index=mut_ids, columns=['p','s','r'])
        # do hydrophobic SASA analysis and average output over WT trajectories
        if 's' in analyses:
            print('analyzing exposed hydrophobic surface..')
            hpsasa, buried = iterate_analysis(self.WT_universes, self.do_sasa_analysis, mut_ids)
            self.WThp_sasa = pd.DataFrame(data=hpsasa, index=mut_ids, columns=['p','s','r'])
            self.WTburied = pd.DataFrame(data=buried, index=mut_ids, columns=['p','s','r'])
        # do other checks and average output over WT trajectories
        if 'c' in analyses:
            print('analyzing structural features..')
            checks, saltb = iterate_analysis(self.WT_universes, self.do_other_checks, mut_ids)
            self.WTchecks = pd.DataFrame(data=checks, index=mut_ids, columns=['c', 'h', 'd'])
            self.WTsaltb = pd.DataFrame(data=saltb, index=mut_ids, columns=['p', 's', 'r'])


    def do_analysis_mut(self, mut_universes, mut_ids=[], analyses='hrsc', reject_trj=True):
        '''
        perform the full sets of analysis on a list of mutant universes, given the mutated resid
        the analyses are: h: do_hbond_analysis(), r: do_rmsf_analysis(), s: do sasa_analysis() c: do_other_checks()
        return the difference of said values compared to the values of the wildtype
        '''
        # reject trajectory with highest rmsd
        output = []
        if reject_trj:
            mut_universes = reject_traj(mut_universes)
        # do hydrogen bond analysis and average output over all trajectories
        if 'h' in analyses:
            print('analyzing hydrogen bonds..')
            e_hbonds, hbonds_unsat = iterate_analysis(mut_universes, self.do_hbond_analysis, mut_ids)
            e_hbonds -= self.WTe_hbonds.loc[mut_ids].values
            hbonds_unsat -= self.WThbonds_unsat.loc[mut_ids].values
            output.extend([e_hbonds, hbonds_unsat])
        # do RMSF analysis and average output over all trajectories
        if 'r' in analyses:
            print('analyzing RMSF of backbone and sidechains..')
            rmsf_bb, score_sc = iterate_analysis(mut_universes, self.do_rmsf_analysis, mut_ids)
            rmsf_bb -= self.WTrmsf_bb.loc[mut_ids].values
            score_sc -= self.WTscore_sc.loc[mut_ids].values
            output.extend([rmsf_bb, score_sc])
        # do sasa analysis and average output over all trajectories
        if 's' in analyses:
            print('analyzing exposed hydrophobic surface..')
            hp_sasa, buried = iterate_analysis(mut_universes, self.do_sasa_analysis, mut_ids)
            hp_sasa -= self.WThp_sasa.loc[mut_ids].values
            output.extend([hp_sasa])
        # do other checks and average output over all trajectories
        if 'c' in analyses:
            print('analyzing structural features..')
            checks, saltb = iterate_analysis(mut_universes, self.do_other_checks, mut_ids)
            checks = np.average(np.array(checks), axis=0) - self.WTchecks.loc[mut_ids].values
            saltb = np.average(np.array(saltb), axis=0) - self.WTsaltb.loc[mut_ids].values
            output.extend([saltb, checks])
        # in case a mutant with multiple mutations is tested, average result of all mutations
        for i, out in enumerate(output):
            if out.shape[0] > 1:
                output[i] = np.average(out, axis=0)
        return np.round(np.concatenate(tuple(output), axis=None),4)

    def do_hbond_analysis(self, universe_in, mut_ids=[], scale_e=False):
        '''
        perform a hydrogen bond analysis. inputs require a universe and the resid of the mutated residue.
        the hydrogen bond analysis is done on 3 selections:
        - the whole protein
        - 8A around the residue
        - just the residue
        It outputs two numbers for each selection:
        - The first output gives the energy of all bonds connecting the selection,both for normal hydrogen 
          bonds and water bridges, giving a total contribution of hydrogen bonds to stability.
        - The second output gives the number of unsatisfied hydrogen bonds, which destabilize the protein.
        '''
        # prepare neccesary selections
        frames = [i.frame for i in universe_in.trajectory]
        prot = universe_in.select_atoms('protein')
        # calculate hydrogen bonds using MDAnalysis
        hbonds_all = HydrogenBondAnalysis(universe=universe_in, between=['protein', 'all'],
                     hydrogens_sel='type H and bonded (type O or type N)',
                     donors_sel = '(type O or type N) and bonded type H',
                     acceptors_sel='type O or (type N and not bonded type H)',
                     d_h_a_angle_cutoff=100, d_a_cutoff = 3.5).run()
        hbonds_all = hbonds_all.results.hbonds
        # get energy for each hydrogen bond
        e_hbonds_all = AnalysisFromFunction(ehbo_yasara, universe_in.trajectory, universe_in, hbonds_all)
        e_hbonds_all.run()
        e_hbonds_all = e_hbonds_all.results.timeseries
        # only keep hbonds where energy > 0
        hbonds_real_ind, e_hbonds_real = np.array([]), np.array([])
        for e_padded in e_hbonds_all:
            e = e_padded[~np.isnan(e_padded)]
            hbonds_real_ind = np.append(hbonds_real_ind, e > 0)
            e_hbonds_real = np.append(e_hbonds_real, e[e>0])
        hbonds_real = hbonds_all[hbonds_real_ind.astype(bool)]
        # do analysis for each given residue
        outputs_e_allmuts = []
        outputs_unsat_allmuts = []
        for resid in mut_ids:
            #print(resid)
            # define the neccesary selections
            resids_s = self.WTsurround_sel.loc[resid].values
            resids_s = resids_s[~np.isnan(resids_s)].astype(int)
            resids_s = ' '.join(list(set(resids_s.astype(str))))
            surround_sel = 'protein and resid {}'.format(resids_s)
            res_sel = 'resid {} and protein'.format(resid)
            selections = ['protein', surround_sel, res_sel]
            # go over each of the selections
            outputs_e, outputs_unsat = [],[]
            for selection in selections:
                e_hbonds = get_energy(universe_in, hbonds_real, e_hbonds_real, selection)
                e_wbridge = get_wbridge(universe_in, hbonds_real, e_hbonds_real, selection)
                e_unsat = get_unsat(universe_in, hbonds_real, selection)
                if scale_e:
                    # scale by max hbond energy of selection
                    hbond_part = get_unsat(universe_in, hbonds_real, selection, return_all=True)
                    max_e = 25*len(hbond_part)/2
                    outputs_e+=[np.average([hb+wb for hb, wb in zip(e_hbonds, e_wbridge)])/max_e]
                    outputs_unsat+=[np.average(e_unsat)/(len(hbond_part)/2)]
                else:
                    outputs_e+=[np.average([hb+wb for hb, wb in zip(e_hbonds, e_wbridge)])]
                    outputs_unsat+=[np.average(e_unsat)]
            outputs_e_allmuts.append(outputs_e)
            outputs_unsat_allmuts.append(outputs_unsat)
        return np.array(outputs_e_allmuts), np.array(outputs_unsat_allmuts)

    def do_rmsf_analysis(self, universe_in, ref_len=1, mut_ids=[]):
        '''
        perform a rmsf analysis. inputs require a universe and the resid of the mutated residue.
        the rmsf analysis is done on 3 selections:
        - the whole protein
        - 8A around the residue
        - just the residue
        It outputs two numbers for each selection:
        - The first output gives the average rmsf of the backbone atoms for each selection
        - The second output gives the average sidechain score for each selection.
          The sidechain score is between 1 to 0 depending on if it is less or more flexible compared
          to the average distribution of amino acids of the same type and solvent exposure.
        '''
        # get average structure and calculate rmsf of backbone
        average = align.AverageStructure(universe_in, universe_in, select='name CA and protein', ref_frame=0).run()
        average = average.results.universe
        aligner = align.AlignTraj(universe_in, average, select='name CA and protein', in_memory=True).run()
        c_alphas = universe_in.select_atoms('name CA and protein')
        rmsf_bb = rms.RMSF(c_alphas).run()
        rmsf_bb = rmsf_bb.results.rmsf
        # calculate rmsf of the sidechains, excluding hydrogen
        rmsf_res = []
        prot = universe_in.select_atoms('protein')
        firstres, lastres = min(prot.residues.resids), max(prot.residues.resids)
        average = align.AverageStructure(universe_in, universe_in, select='protein', ref_frame=0).run()
        average = average.results.universe
        counter = 2*ref_len+1
        for res in prot.residues:
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
            #print(res.resname, rmsf)
            counter+=1
        # classify each sidechain based on surface exposure and AA type
        sasa_dict = get_sasa_class(prot)
        res_class = [res.resname+'-'+sasa_dict[res.resid] for res in prot.residues]
        # turn rmsf of sidechain into score by comparing to curve of expected values
        res_score = []
        for rmsf, res_cl in zip(rmsf_res, res_class):
            res_score.append(dens_to_score(rmsf, *find_xy_dens(self.rmsf_range, res_cl))[0])
        res_score = np.array(res_score)
        # turn into a dataframe to access the right values using resids
        rmsf_bb_out, score_sc_out = [], []
        df_rmsf_bb = pd.DataFrame(data=rmsf_bb, index=prot.residues.resids)
        df_score_sc = pd.DataFrame(data=res_score, index=prot.residues.resids)
        # for reach residue return average values for the protein, surrounding, and residue
        for resid in mut_ids:
            rmsf_bb_res, score_sc_res = [], []
            resids_s = self.WTsurround_sel.loc[resid].values
            resids_s = resids_s[~np.isnan(resids_s)].astype(int)
            surround_ids = list(set(resids_s))
            selections = [prot.residues.resids, surround_ids, resid]
            for selection in selections:
                rmsf_bb_sel = np.average(df_rmsf_bb.loc[selection].values)
                score_sc_sel = np.average(df_score_sc.loc[selection].values)
                rmsf_bb_res.append(rmsf_bb_sel)
                score_sc_res.append(score_sc_sel)
            rmsf_bb_out.append(rmsf_bb_res)
            score_sc_out.append(score_sc_res)
        return np.array(rmsf_bb_out), np.array(score_sc_out)

    def do_sasa_analysis(self, universe_in, mut_ids=[]):
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
        # get libraries to switch between three letter and single letter form
        switch_AA = {'R':'ARG', 'H':'HIS', 'K':'LYS', 'D':'ASP', 'E':'GLU',
             'S':'SER', 'T':'THR', 'N':'ASN', 'Q':'GLN', 'C':'CYS',
             'G':'GLY', 'P':'PRO', 'A':'ALA', 'V':'VAL', 'I':'ILE',
             'L':'LEU', 'M':'MET', 'F':'PHE', 'Y':'TYR', 'W':'TRP'}
        rswitch_AA = {v: k for k, v in switch_AA.items()}
        hp_sel = '(type C or (type H and bonded type C))'
        prot = universe_in.select_atoms('protein')
        # get df with sasa for entire protein
        hp_cols = get_sel_sasa(prot, hp_sel).columns
        sasa_prot = AnalysisFromFunction(get_sel_sasa, universe_in.trajectory, prot, hp_sel).run()
        df_sasa_prot = pd.DataFrame(data=np.average(sasa_prot.results.timeseries, axis=0), columns=hp_cols)
        resids = list(set(prot.residues.resids))
        resids.sort()
        sasa_score_r = []
        buried_count_r = []
        # get sasa score per residue
        for i in resids:
            res_sel = 'resid {} and protein'.format(i)
            res_atm = prot.select_atoms('({}) and ({})'.format(res_sel, hp_sel))
            # record total sasa of residue
            sasa_r_resid = np.sum(df_sasa_prot[res_atm.atoms.indices].values)/len(res_atm.residues)
            # count number of buried hydrophobic atoms
            buried_count_r.append(np.sum(df_sasa_prot[res_atm.atoms.indices].values == 0)/len(res_atm.residues))

            resname = list(set(res_atm.residues.resnames))
            if len(resname) > 1:
                sasa_score_r.append(np.array([0]))
                #print('warning: for resid {}, more than one residue type was found'.format(i))
                continue

            sasa_dens_x = self.sasa_range.columns.values.astype(float)
            sasa_dens_y = self.sasa_range.loc[rswitch_AA[resname[0]]].values
            sasa_score_r_resid = dens_to_score(sasa_r_resid, sasa_dens_x, sasa_dens_y)
            sasa_score_r.append(sasa_score_r_resid)
        # for each residue return average values for the protein, surrounding, and residue
        sasa_score_out, buried_count_out = [], []
        sasa_score_r = pd.DataFrame(data=np.array(sasa_score_r).reshape(1, len(sasa_score_r)), columns=resids)
        buried_count_r = pd.DataFrame(data=np.array(buried_count_r).reshape(1, len(buried_count_r)), columns=resids)
        for resid in mut_ids:
            resids_p = list(set(prot.residues.resids))
            resids_s = self.WTsurround_sel.loc[resid].values
            resids_s = resids_s[~np.isnan(resids_s)].astype(int)
            resids_s = list(set(resids_s))
            selections = [resids_p, resids_s, resid]
            sasa_score_res, buried_count_res = [], []
            for selection in selections:
                sasa_score_res.append(np.average(sasa_score_r[selection].values))
                buried_count_res.append(np.sum(buried_count_r[selection].values))
            sasa_score_out.append(sasa_score_res)
            buried_count_out.append(buried_count_res)
        return np.array(sasa_score_out), np.array(buried_count_out)

    def do_other_checks(self, universe_in, mut_ids=[]):
        '''
        Does various other checks not included in the analyses done above. Three checks are done 
        just on the whole protein, instead of all three selections. These include: 
        - a check for the number of helix capping motifs
        - a check on the number of prolines inside the alpha helix
        - a check on the amount of surface exposed sulphurs that might oxidize
        - a check on whether the number of disulfide bridges has changed
        Besidse these four, a third analysis is done on the number of saltbridges.
        This analysis is done on all three selections (protein, surrounding, residue)
        '''
        protWT = self.WT_universes[0].select_atoms('protein')
        prot = universe_in.select_atoms('protein')
        # find helix capping motifs, just take first WT universe for structure check
        starts_wt, ends_wt = get_helixcaps(protWT, output='index')
        starts_mut, ends_mut = get_helixcaps(prot, starts=starts_wt, ends=ends_wt, output='index')
        capcount = len(starts_mut)+len(ends_mut)
        # calculate helix score for each res in helix, just take first WT universe for structure check
        secstr = get_pydssp(protWT)
        resnames = prot.residues.resnames
        helix_score = get_helix_score(resnames, secstr)
        # check for exposed sulfur groups
        #sulphur_sasa = AnalysisFromFunction(get_sel_sasa, universe_in.trajectory,
        #                                    prot, 'type S or (type H and bonded type S)').run()
        #sulphur_sasa = sulphur_sasa.results['timeseries']
        #check for disulfide bridges
        disulfide = get_disulfide(universe_in)
        #combine checks
        checks_out = [np.average(capcount), np.average(helix_score), disulfide]
        checks_out = len(mut_ids)*[checks_out]
        # go over each residue and get saltbridges for protein, surrounding and residue
        saltbridges_out =[]
        for resid in mut_ids:
            saltbridges_res = []
            # define the neccesary selections
            resids_s = self.WTsurround_sel.loc[resid].values
            resids_s = resids_s[~np.isnan(resids_s)].astype(int)
            resids_s = ' '.join(list(set(resids_s.astype(str))))
            surround_sel = 'protein and resid {}'.format(resids_s)
            res_sel = 'resid {} and protein'.format(resid)
            selections = ['protein', surround_sel, res_sel]
            for selection in selections:
                # check for salt bridges
                saltbridges = AnalysisFromFunction(get_saltbridge, universe_in.trajectory, prot, selection)
                saltbridges.run()
                saltbridges = saltbridges.results['timeseries']
                saltbridges_res.append(np.average(saltbridges))
            saltbridges_out.append(saltbridges_res)
        return np.array(checks_out), np.array(saltbridges_out)
