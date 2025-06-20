import argparse
import warnings
from .BoostMut_class import BoostMut
from .utils import *


def main():
    parser = argparse.ArgumentParser(prog='BoostMut',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    description='script for analyzing short high-throughput MD trajectories using MDAnalysis')
    basic_group = parser.add_argument_group('Basic Arguments')
    basic_group.add_argument('-i', '--inputdir', required=True, type=str, help='input directory containing subdirectories with trajectories for each mutation')
    basic_group.add_argument('-o', '--output', default='BoostMut_out.csv', help='name of the output .csv')
    basic_group.add_argument('-m', '--mutfile', default="", help='file containing a list of mutations to analyze, if kept default, will analyze all mutations in input directory')
    #basic_group.add_argument('-a', '--analysis', default='hrsc', help='reported analysis in output h:hydrogen bond analysis, r:rmsf analysis, s:sasa analysis, c:other structural checks ')
    basic_group.add_argument('-s', '--selection', default=['hbse:sr', 'c:p'], nargs='+', help='reported selections per analysis, p: whole protein, s:surrounding of mutation, r:just the mutation')
    basic_group.add_argument('-t', '--time', default='50', help='length of the trajectory in picoseconds')

    advanced_group = parser.add_argument_group('Advanced Arguments')
    advanced_group.add_argument('-n1', '--wtname', default='Subdir_template', help='subdirectory for the wildtype')
    advanced_group.add_argument('-n2', '--mutname', default='Subdir_[A-Z][0-9]+[A-Z]', help='regex the subdirectories for each mutation has to satisfy')
    advanced_group.add_argument('-n3', '--topname', default='^[\w\d].*bonds\.pdb$', help='regex each of the topology files has to satisfy')
    advanced_group.add_argument('-n4', '--trajname', default='^[\w\d].*\.xtc$', help='regex each of the trajectory files has to satify')
    advanced_group.add_argument('-n5', '--bondsname', default="", help='regex for seperate files with bondinfo if the topologies do not contain it')
    advanced_group.add_argument('-gb', '--guessbonds', default=False, help='lets MDAnalysis guess bonds, making the calculations significantly slower')
    advanced_group.add_argument('-sf', '--sasafile', help='name of custom file containing the benchmarks of residue sasa located in benchmarks')
    advanced_group.add_argument('-rf', '--rmsffile', help='name of custom file containing the benchmarks for sidechain rmsf located in benchmarks')
    advanced_group.add_argument('-rs', '--rangesur', default=8, help='range around the mutation used in the surrounding selection in Å')
    advanced_group.add_argument('-rt', '--rejecttraj', default=True, help='if set to True, rejects the trajectory with highest RMSD for each mutation')
    advanced_group.add_argument('-lc', '--lastcheck', default="", help='filename of the .csv from the last checkpoint from which to continue')
    advanced_group.add_argument('-cp', '--checkpoint', default=True, help='if set to True, saves the result after each mutation')
    args = parser.parse_args()

    # exclude mutations already analyzed in .csv given by --lastcheck
    if len(args.lastcheck) > 0:
        df_done = pd.read_csv(args.lastcheck, index_col=0)
        exclude = df_done.index.values
        print('already analyzed:', exclude)
    else:
        exclude = []
    # get paths for wildtype and mutants, and positions of all selected mutations
    wt_path = os.path.join(args.inputdir, args.wtname)
    muts, mutpos, mut_paths  = get_mutinfo(args.inputdir, args.mutname, args.mutfile, exclude=exclude)

    # make sure all neccesary files are present
    print('analyzing:', muts)
    if not os.path.isdir(wt_path):
        raise Exception('could not find the WT directory')
    if len(mut_paths) == 0 and len(args.lastcheck) > 0:
         df_scaled = scale_df(df_done)
         df_scaled.to_csv(os.path.splitext(args.output)[0]+'_scaled.csv')
         raise Exception('no mutant directories found using regex pattern')
    elif len(mut_paths) == 0:
        raise Exception('no mutant directories found using regex pattern')

    # get column names of output df
    cols_full = get_columnnames(selection=['hbsec:psr'])
    cols_sel = get_columnnames(selection=args.selection)
    print(cols_full)
    print(cols_sel)
    if len(args.lastcheck) > 0:
        if cols_sel != list(df_done.columns):
            raise Exception('Columns do not match last checkpoint, make sure the same analyses and selections are used')

    # get the right benchmark curves, overwrite specified time if custom benchmark files are provided
    if args.sasafile == None:
        sasa_file = 'range_sasa_{}ps.csv'.format(args.time)
    else:
        sasa_file = args.sasafile
    if args.rmsffile == None:
        rmsf_file = 'range_rmsf_{}ps.csv'.format(args.time)
    else:
        rmsf_file = args.rmsffile
    print('using benchmark files:', sasa_file, rmsf_file)

    # load WT trajectories into MDAnalysis and initialize BoostMut with the WT analysis
    wt_universes = load_universes(wt_path, topname=args.topname, trajname=args.trajname, bondstabname=args.bondsname, guess_bonds=args.guessbonds)
    boostmut = BoostMut(wt_universes, mut_ids=sorted(list(set(mutpos))), rmsf_loc=rmsf_file, sasa_loc=sasa_file, reject_trj=args.rejecttraj)
    boostmut.do_analysis_WT(mut_ids=sorted(list(set(mutpos))), analyses=args.analysis)

    # go over each of the mutations and analyze
    data_out, index_out = [], []
    for mut, pos, path in zip(muts, mutpos, mut_paths):
        print(mut, pos, path)
        mut_universes = load_universes(path, topname=args.topname, trajname=args.trajname, bondstabname=args.bondsname, guess_bonds=args.guessbonds)
        data_mut = boostmut.do_analysis_mut(mut_universes, mut_ids=[pos], analyses=args.analysis, reject_trj=args.rejecttraj)
        index_out.append(mut)
        data_out.append(data_mut)
        if args.checkpoint:
            data_arr = np.vstack(data_out)
            df_out_full = pd.DataFrame(data=data_arr, columns=cols_full, index=index_out)
            df_out_sel = df_out_full[cols_sel]
            if len(args.lastcheck) > 0:
                df_out_sel = pd.concat((df_done, df_out_sel))
            df_out_sel.to_csv(args.output)
    # process final output
    data_arr = np.vstack(data_out)
    df_out_full = pd.DataFrame(data=data_arr, columns=cols_full, index=index_out)
    df_out_sel = df_out_full[cols_sel]
    if len(args.lastcheck) > 0:
        df_out_sel = pd.concat((df_done, df_out_sel))
    df_out_sel.to_csv(args.output)
    df_out_scaled = scale_df(df_out_sel)
    df_out_scaled.to_csv(os.path.splitext(args.output)[0]+'_scaled.csv')



if __name__ == "__main__":
    main()
