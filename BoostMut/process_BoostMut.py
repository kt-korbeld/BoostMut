import os
import pandas
import warnings
import argparse
from .utils import *
from .mk_benchmarks import *

def main():
    parser = argparse.ArgumentParser(prog='BoostMut',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    description='script for processing the output of BoostMut')
    subparsers = parser.add_subparsers(
                     title="processing tools",
                     dest="command",
                     help="available processing tools")

    # parser to combine outputs into one .csv
    combine_parser = subparsers.add_parser("combine",help="combine seperate outputs or data from primary predictors")
    combine_parser.add_argument('-i', '--input', type=str, nargs='+', required=True, help='path of one or more .csv files to be combined')
    combine_parser.add_argument('-o', '--output', type=str, default='BoostMut_combined.csv', help='name of combined output .csv file')
    combine_parser.add_argument('-m', '--add_metric', type=str, nargs='+', default=[], help='one or more .csv files with scores of primary predictor or other metrics to add to the BoostMut evaluation')
    combine_parser.add_argument('-n', '--metric_name', type=str, nargs='+', default=['primary_pred'], help='column names of new metrics added in output if add_metric is specified')
    combine_parser.add_argument('-c', '--metric_col', type=int, nargs='+', default=[1], help='column numbers to use in each .csv file specified in add_metric')

    # parser to scale output
    scale_parser = subparsers.add_parser("scale", help="scale given BoostMut output by its standard deviation")
    scale_parser.add_argument('-i', '--input', type=str, help='name of raw BoostMut output .csv file to scale')
    scale_parser.add_argument('-o', '--output', type=str, default='BoostMut_out_scaled.csv', help='name of scaled output .csv file')
    scale_parser.add_argument('-n', '--metric_name', type=str, nargs='+',default=[],  help='if additional metrics were added, specify column names of metrics')
    scale_parser.add_argument('-s', '--metric_scale', type=int, nargs='+', default=[-1],  help='if additional metrics were added, specify if high scores are desirable (1) or undesirable (-1)')
    scale_parser.add_argument('-e', '--exclude_metric', type=str, nargs='+', default=[], help='metrics to exclude from total score')

    # parser to create excel output
    excel_parser = subparsers.add_parser("excel", help="convert .csv output into a human-readable excel file")
    excel_parser.add_argument('-i', '--input', type=str, help='path of .csv with BoostMut outputs to be converted into a .xlsx file')
    excel_parser.add_argument('-o', '--output', type=str, help='name of the output .xlsx file')

    # parser to create new benchmarks
    bench_parser = subparsers.add_parser('benchmark', help='create new benchmarks for BoostMut')
    bench_parser.add_argument('-i', '--inputdir', type=str, help='input directory containing subdirectories with trajectories for each mutation')
    bench_parser.add_argument('-o', '--output', type=str, default='benchmark_new.csv', help='output .csv file with new benchmark curves')
    bench_parser.add_argument('-m', '--mode', type=str, default='rmsf', help='generate rmsf or sasa benchmarks')
    bench_parser.add_argument('-n1', '--topname', default='^[\w\d].*\.tpr$', help='regex each of the topology files has to satisfy')
    bench_parser.add_argument('-n2', '--trajname', default='^[\w\d].*\.xtc$', help='regex each of the trajectory files has to satify')
    args = parser.parse_args()

    if args.command == "combine":
        # load in and combine dataframes
        print(args.input)
        if args.input == None:
            raise ValueError('No input specified')
        dfs_set = [pd.read_csv(i, index_col=0) for i in args.input]
        df_combined = combine_df(dfs_set)
        # combine with additional scores if provided
        if len(args.add_metric) != 0:
            df_combined = add_predictors(df_combined, args.add_metric, args.metric_col, args.metric_name)
        df_combined.to_csv(args.output)
        print('output saved as ', args.output)

    if args.command == "scale":
        print(args.input)
        if args.input == None:
            raise ValueError('No input specified')
        # load dataframe, add custom scaling if specified
        df_in = pd.read_csv(args.input, index_col=0)
        custom_scale = get_custom_scale(df_in, args.metric_name, args.metric_scale)
        df_scaled = scale_df(df_in, scale_custom=custom_scale)
        # if specified, exclude metrics from total score
        if len(args.exclude_metric) > 0:
            print('excluding {} from total score'.format(args.exclude_metric))
        cols_mean_score = [i for i in df_scaled.columns if not i in args.exclude_metric]
        df_scaled['mean'] = np.round(df_scaled[cols_mean_score].mean(axis=1).values, 4)
        df_scaled.to_csv(args.output)
        print('output saved as ', args.output)

    if args.command == "excel":
        print(args.input)
        if args.input == None:
            raise ValueError('No input specified')
        df_in = pd.read_csv(args.input, index_col=0)
        if args.output == None:
            generate_excel(df_in, excel_out=args.input)
        else:
            generate_excel(df_in, excel_out=args.output)

    if args.command == 'benchmark':
        if args.inputdir == None:
            raise ValueError('No input directory specified')
        if args.mode == 'rmsf':
            print('making new rmsf benchmark with the name: ', args.output)
            make_benchmark_rmsf(args.inputdir, args.trajname, args.topname, args.output)
        elif args.mode == 'sasa':
            print('making new sasa benchmark with the name: ', args.output)
            make_benchmark_sasa(args.inputdir, args.trajname, args.topname, args.output)
        else:
           raise ValueError("Mode should be 'sasa' or 'rmsf'")


if __name__ == "__main__":
    main()
