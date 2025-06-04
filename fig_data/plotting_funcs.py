import os
from functools import partial 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import auc
from scipy.stats import spearmanr
from matplotlib_venn import venn3
import secstructartist as ssa

def plot_spearman(data, name_out='out-spearman.png'):
    '''
    plot spearman correlation with final datapoint to show convergence of ranking
    '''
    times = np.arange(0,1050,50)
    for i, timeseries in enumerate(data):
        sprmn = [spearmanr(timeseries[:,i], timeseries[:,-1])[0] for i in range(timeseries.shape[-1]) ]
        # define colors and names
        if i == 0:
            c = 'firebrick'
            l = 'whole protein'
        if i == 1:
            c = 'mediumvioletred'
            l = '8Å surrounding'
        if i == 2:
            c='tab:blue'
            l = 'residue'
        plt.plot(times, sprmn, color=c, label=l)
        plt.scatter(times, sprmn, color=c, s=10)
    plt.yscale('function', functions=(partial(np.power, 10.0), np.log10))
    plt.grid(True, which="major", alpha=0.6)
    plt.xticks(list(plt.xticks()[0]) + [50])
    plt.xlabel('time (ps)')
    plt.ylabel('Spearman ρ with 1ns')
    plt.xlim(0,1000)
    plt.ylim(0,1.01)
    plt.legend()
    plt.savefig(name_out, dpi=300, bbox_inches='tight')
    plt.show()

def plot_3dhist(data, highlight=[],name_out='out-3dhist.png'):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(projection='3d')

    #generate consistent bins
    a, b = np.histogram(data['1000'].values, bins=30)
    count = 0
    for time in data.columns:
        count+=1
        c = count/len(data.columns)
        sel = [i for i in data[time].index if  i != 'N92K']
        data_t = data.loc[sel][time].values

        a, b = np.histogram(data_t, bins=b)
        bx = (b[1:] + b[:-1]) / 2
        bw = abs(np.average(b[1:] - b[:-1]))    

        a2, b2 = np.histogram(data[time].loc[highlight], bins=b)
        bx2 = (b2[1:] + b2[:-1]) / 2
        # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
        ax.bar(bx2, a2, zs=int(time), zdir='y', color='green', alpha=1, width=bw, lw=1, ec='black')
        ax.bar(bx, a, zs=int(time), zdir='y', color=(c,0,1-c), alpha=0.8, width=bw, lw=1, ec='black')
        ax.bar(bx2, a2, zs=int(time), zdir='y', color='green', alpha=1, width=bw, lw=1, ec='black')

    ax.set_xlabel('mean BoostMut score')
    ax.set_ylabel('time (ps)')
    plt.savefig(name_out, bbox_inches='tight', dpi=300)
    plt.show()    
    
def roc_normal(in_true, in_vals):
    '''
    standard ROC curve
    '''
    in_true = np.array(in_true)
    in_vals = np.array(in_vals)
    # get total number of true positives and negatives
    total_pos = len([i for i in in_true if i == 1])
    total_neg = len([i for i in in_true if i == 0])
    # sort positives/negatives according to given values
    true_sorted = in_true[np.argsort(in_vals)[::-1]]
    #generate positions on ROC curve for each value in sorted list
    roc_list_x, roc_list_y = [0],[0]
    roc_val_x, roc_val_y = 0, 0 
    for i in true_sorted:
        if i == 1: 
            roc_val_y += 1/total_pos
        if i == 0:
            roc_val_x += 1/total_neg
        roc_list_x.append(roc_val_x)
        roc_list_y.append(roc_val_y)
    return roc_list_x, roc_list_y


def roc_tm(in_true, in_vals):
    '''
    adapted version of ROC curve that scales positives by their relative contribution
    '''
    vals_true = np.array(in_true)
    vals_true = vals_true / sum(vals_true[vals_true > 0])
    in_vals = np.array(in_vals)
    total_neg = len([i for i in vals_true if i == 0])
    true_sorted = vals_true[np.argsort(in_vals)[::-1]]
    roc_list_x, roc_list_y = [0],[0]
    roc_val_x, roc_val_y = 0, 0 
    for i in true_sorted:
        if i > 0: 
                roc_val_y += i
        else:
            roc_val_x += 1/len(vals_true[vals_true <= 0])
        roc_list_x.append(roc_val_x)
        roc_list_y.append(roc_val_y)
    return roc_list_x, roc_list_y

def plot_roc(data, scale=False, exclude=[], name_out='out-roc.png'):
    '''
    plot the ROC curves for a given dataset
    '''
    aucs = []
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(5, 5))
    boostmut_cols = [i for i in data.columns if i not in ['exp_tested', 'stabilizing', 'dtm']+exclude]
    # only take experimentally tested values
    data_exp = data[data.exp_tested]
    # the foldx values with normalized stdev are used normalization does not affect the ROC curve
    data_pred_fx = data_exp['foldx'].values
    data_pred_bm = data_exp[boostmut_cols].sum(axis=1).values
    if scale:
        data_true = [0 if np.isnan(i) else i for i in data_exp.dtm.values.astype(float)]
        x1,y1 = roc_tm(data_true, data_pred_fx)
        x2,y2 = roc_tm(data_true, data_pred_bm)
    else:
        data_true = data_exp.stabilizing.values.astype(int)
        x1,y1 = roc_normal(data_true, data_pred_fx)
        x2,y2 = roc_normal(data_true, data_pred_bm)
    plt.plot(x1,y1, label='FoldX')
    plt.plot(x2,y2, label='FoldX+BM', color='firebrick')
    aucs.append(auc(x1,y1))
    aucs.append(auc(x2,y2))  
    plt.plot(np.arange(0,1.2,0.1),np.arange(0,1.2,0.1), color='grey', alpha=0.5)
    plt.legend(loc=2)
    plt.xlim(-0.02,1)
    plt.ylim(0,1.02)
    plt.savefig(name_out, bbox_inches='tight', dpi=300)
    plt.show()
    return aucs

def plot_roc_fullrange(data, scale=False, exclude=[], 
                       predictor='foldx_raw', predictor_scaled='foldx', cutoff=0, 
                       name_out='out-roc-full.png'):
    '''
    plot the ROC curves for a given dataset where both mutations passing and not passing
    the primary predictor are included.
    '''
    aucs = []
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(5, 5))
    # get index of muts passing and not passing cutoff
    boostmut_cols = [i for i in data.columns if i not in ['foldx_raw', 'ddg']+exclude]
    index_passed = [i for i in data.index if data[predictor].loc[i] <= cutoff]
    index_notpassed = [i for i in data.index if data[predictor].loc[i] > cutoff]
    index_complete = np.append(index_passed,index_notpassed)
    # sort all values passing cutoff before all all values not passing cutoff 
    vals_fx = data.loc[index_complete][predictor_scaled].values
    vals_passed = data.loc[index_passed][boostmut_cols].sum(axis=1).values + 1000
    vals_notpassed = data.loc[index_notpassed][predictor_scaled].values - 1000
    vals_complete = np.append(vals_passed, vals_notpassed)
    
    # get true vals and location of cutoff in ROC curve
    if scale:
        data_true = [ max(0, -i) for i in data.loc[index_complete].ddg.values]
        cutoff_x = len([i for i in data_true[:len(index_passed)] if not i]) / len([i for i in data_true if not i])
        x1,y1 = roc_tm(data_true, vals_fx)
        x2,y2 = roc_tm(data_true, vals_complete)
    else:
        data_true = [ i < 0 for i in data.loc[index_complete].ddg.values] 
        cutoff_x = len([i for i in data_true[:len(index_passed)] if i == 0]) / len([i for i in data_true if i == 0])
        x1,y1 = roc_normal(data_true, vals_fx)
        x2,y2 = roc_normal(data_true, vals_complete)
    pred_names = {'foldx':'FoldX','SO':'Stability Oracle'}
    plt.plot(x1,y1, label='{}'.format(pred_names[predictor_scaled]), color='tab:blue')
    plt.plot(x2,y2, label='{}+BM'.format(pred_names[predictor_scaled]), color='firebrick')
    aucs.append(auc(x1,y1))
    aucs.append(auc(x2,y2))  
    plt.axvspan(cutoff_x, 1, color='grey', alpha=0.3, lw=0)
    plt.legend(loc='lower right')
    plt.xlim(-0.02,1)
    plt.ylim(0,1.02)
    plt.savefig(name_out, bbox_inches='tight', dpi=300)
    plt.show()     
    return aucs    
        
def plot_venn(data, name_out='out-venn.png', add_stab=[], updated=False):
    '''
    plot a venn diagram of experimentally tested mutations vs 
    an identically sized selection based on BoostMut scores
    '''
    boostmut_cols = [i for i in data.columns if i not in ['exp_tested', 'stabilizing', 'dtm']]
    sel_exp =  data[data.exp_tested].index
    sel_boostmut = data[boostmut_cols].sum(axis=1).sort_values(ascending=False).index[:len(sel_exp)]
    sel_stab = list(data[data.stabilizing].index)+list(add_stab)
    
    v = venn3([set(sel_boostmut), set(sel_exp), set(sel_stab)], ['BoostMut', 'Visual', ''],
         set_colors=("grey", "tab:blue", "firebrick"))
    v.get_patch_by_id('100').set_color('grey')
    v.get_patch_by_id('010').set_alpha(0.8)
    v.get_patch_by_id('011').set_alpha(1)
    v.get_patch_by_id('100').set_linestyle('dashed')
    v.get_patch_by_id('110').set_alpha(1)
    v.get_patch_by_id('110').set_color('tab:blue')
    v.get_patch_by_id('110').set_ec('black')
    v.get_patch_by_id('110').set_lw(1)
    v.get_patch_by_id('111').set_alpha(1)
    v.get_patch_by_id('111').set_color('firebrick')
    v.get_patch_by_id('111').set_ec('black')
    v.get_patch_by_id('111').set_lw(1)
    #new
    if updated:
        v.get_patch_by_id('001').set_color('white')
        v.get_label_by_id('001').set_text('')
        v.get_patch_by_id('101').set_ec('black')
        v.get_patch_by_id('101').set_alpha(1)
        v.get_patch_by_id('101').set_color('#C87673')
        v.get_patch_by_id('101').set_lw(1)
        v.get_patch_by_id('101').set_ec('black')
    #v.get_patch_by_id('100').set_color('dashed')
    plt.savefig(name_out, bbox_inches='tight', dpi=300)
    plt.show()

def plot_bars(values, species, colors, name_out='out-bar.png', headspace=0.14):
    '''
    plot barplot with weighted and unweighted AUC values
    '''
    plt.rcParams.update({'font.size': 14})
    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    fig.set_figwidth(4)
    for attribute, measurement in values.items():
        offset = width * multiplier +0.12
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[attribute])
        print(measurement)
        ax.bar_label(rects, padding=5, rotation=70)
        multiplier += 1
    ax.set_ylabel('AUC')
    ax.set_xticks(x + width, species)
    ax.legend(loc='upper left',)
    ax.set_ylim(0., 1+headspace)
    ax.set_xlim(-0.2, 2.7)
    plt.hlines(0.5, -0.5,3, color='red', alpha=0.5, zorder=-2, linestyle='dashed')
    plt.savefig(name_out, bbox_inches='tight', dpi=300)
    plt.show()

def plot_square_area(ax, numbers_in, x_offset=0, x_size=1, y_size=1, colors=('grey','tab:blue', 'firebrick')):
    '''
    plot a square treemap-inspired plot showing the ratio between 3 given numbers
    '''
    # prepare neccesary variables
    u_c, s_c, d_c = colors
    untest, stab, destab = numbers_in
    total = untest+destab+stab
    stabrat = stab/(destab+stab)
    x_div = x_size*(untest/total)+x_offset
    y_div = y_size*stabrat
    
    # get tuple of x, y1, y2 coordinates for each area 
    untest_area = ([x_offset, x_div], [0,0], [y_size,y_size])
    stab_area = ([x_div, x_size+x_offset], [0,0], [y_div, y_div])
    destab_area = ([x_div, x_size+x_offset], [y_div, y_div], [y_size,y_size])
    
    # fill area according to color
    ax.fill_between(*untest_area, color=u_c, alpha=0.7 )
    ax.fill_between(*destab_area, color=s_c)
    ax.fill_between(*stab_area, color=d_c)

    if int(untest) != 0:
        ax.text(0.5*x_div, 0.5*y_size, str(int(untest)), horizontalalignment='center', verticalalignment='center')
    if int(destab) != 0:
        ax.text(x_div+0.5*(x_size-x_div+x_offset), y_div+0.5*(y_size-y_div), str(int(destab)), horizontalalignment='center', verticalalignment='center')
    if int(stab) != 0:
        ax.text(x_div+0.5*(x_size-x_div+x_offset), 0.5*(y_div), str(int(stab)), horizontalalignment='center', verticalalignment='center')

def plot_square(data, name_out='out-square.png'):
    '''
     plot a square treemap-inspired plot showing the fraction of stabilizing mutations
    '''
    boostmut_cols = [i for i in data.columns if i not in ['exp_tested', 'stabilizing', 'dtm']]
    sel_exp =  set(data[data.exp_tested].index)
    sel_boostmut = set(data[boostmut_cols].sum(axis=1).sort_values(ascending=False).index[:len(sel_exp)])
    sel_stab = set(data[data.stabilizing].index)
    
    a1, b1, c1 = len(sel_boostmut-sel_exp),  len(sel_boostmut&sel_stab), len(sel_boostmut&sel_exp-sel_stab)
    a2, b2, c2 = 0, len(sel_stab), len(sel_exp-sel_stab)
    
    fig, ax = plt.subplots()
    fig.set_figheight(3)
    plot_square_area(ax, (a1, b1, c1))
    plot_square_area(ax, (a2, b2, c2), x_offset=1.2)
    ax.axis('off')
    plt.savefig(name_out, bbox_inches='tight', dpi=300)
    plt.show()

def plot_new_tms(ranking, name_out='out-ranking.png'):
    '''
    plot a barplot showing the dTms of all new mutations,
    sorted by BoostMut ranking. 
    '''
    # some functions needed to map second x axis to first
    def name2rank(n):
        return ranking_c.loc[n]['auto_rank'].values
    def rank2name(r):
        temp = ranking_c.copy()
        temp['name'] = temp.index
        temp.index = temp['auto_rank']
        return temp.loc[r].name.values
    
    ranking_c = ranking.copy()
    #leh_ranking = leh_ranking_df.index
    ranking_c = ranking_c.sort_values(by='auto_rank')

    fig, ax = plt.subplots()
    ax.bar(ranking_c.index.values, ranking_c.dTm.values, width=1, lw=1, ec='black')
    ax.set_xlabel('Mutation')
    #ax.secondary_xaxis('top', functions=(rank2name, name2rank))
    plt.xticks(rotation=45, ha='center')
    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xbound(ax.get_xbound())
    ax2.set_xticklabels(name2rank(ranking_c.index.values).astype(str))
    ax2.set_xlabel('Rank')
    plt.xticks(rotation=45, ha='center')
    #ax.secondary_xaxis('top', newtms_df.auto_rank.values.astype(str))
    ax.set_ylabel('ΔTm (C)')
    plt.savefig(name_out, bbox_inches='tight', dpi=300)
    plt.show()

def plot_mut_distribution(res_start, prot_secstr, res_stab_old, res_stab_new, name_out='out-mut-dist.png'):
    '''
    plot distribution of new mutations over the length of the protein, 
    plotting the secondary structure of the protein below the plot.
    '''
    plt.rcParams.update({'font.size': 16})
    protlength = len(prot_secstr)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8,5), dpi=96, sharex=True,
        gridspec_kw={"height_ratios": [10,1], "hspace": .01})
    
    ax0.hist(res_stab_new, bins=np.arange(res_start,protlength+res_start,1),
             alpha=1, lw=0.5, ec='black', label='new mutations', color='firebrick')
    ax0.hist(res_stab_old, bins=np.arange(res_start,protlength+res_start,1),
             alpha=1, lw=0.5, ec='black', label='original FRESCO mutations', color='darkgrey')
    artist = ssa.SecStructArtist()
    artist["H"].fillcolor = ('skyblue')
    artist["H"].shadecolor = ('cornflowerblue')
    artist["S"].fillcolor = ('firebrick')
    artist.draw(prot_secstr, xpos=list(range(5, 5+len(prot_secstr))),ax=ax1)

    ax0.yaxis.set_ticks([0,1,2,3,4])
    ax0.legend()
    ax1.yaxis.set_ticklabels([])
    ax1.yaxis.set_ticks([])
    ax1.set_xlabel('residue index')
    plt.savefig(name_out, bbox_inches='tight', dpi=300)
    plt.show()

def process_raw_benchstab(benchstab_data, predictors, mut_index):
    '''
    convert raw benchstab output into a dataframe with ddG for each predictor in one column
    '''
    df_predscores = {pred:[] for pred in predictors}
    
    for pred in predictors:
        data_pred = benchstab_data[benchstab_data.predictor == pred].copy()
        data_pred.index = data_pred.mutation.values
        data_pred = data_pred.loc[mut_index]
        df_predscores[pred] = data_pred.DDG.values
    df_predscores = pd.DataFrame(df_predscores, index=mut_index)
    return df_predscores

def get_minmax_roc(x_vals, y_true, y_unknown, return_auc=False):
    y_true = np.array(y_true)
    y_unknown = np.array(y_unknown).astype(int)
    maxauc, minauc = 0, 1
    max_x, max_y = [], []
    min_x, min_y = [], []
    for ind in range(1,len(y_true)):
        # for increasing both forward and reverse, set all uknowns to 1
        selected_f = np.append(np.ones(ind), np.zeros(len(y_true)-ind))
        selected_r = np.append(np.zeros(len(y_true)-ind), np.ones(ind))
        
        true_temp_f, true_temp_r = y_true.copy(), y_true.copy()
        true_temp_f[np.logical_and(selected_f, y_unknown)] = 1
        true_temp_r[np.logical_and(selected_r, y_unknown)] = 1
        # get auc for each new option, set to auc
        xf,yf = roc_normal(true_temp_f, x_vals)
        xr,yr = roc_normal(true_temp_r, x_vals)
        auc_f, auc_r = auc(xf,yf), auc(xr,yr)
        if auc_f > maxauc:
            maxauc = auc_f
            max_x, max_y = xf, yf    
        if auc_r < minauc:
            minauc = auc_r
            min_x, min_y = xr, yr
    if return_auc:
        return maxauc, minauc
    else:
        return max_x, max_y, min_x, min_y

def plot_range_roc(x1, y1, x2, y2, stepsize=1000, color='red'):
    x_common = np.linspace(0, 1, stepsize)
    # Interpolate the y values for both curves on the common x-axis points
    y1_interp = np.interp(x_common, x1, y1)
    y2_interp = np.interp(x_common, x2, y2)
    # plot two ranges and fill inbetween
    plt.plot(x1, y1, color=color)
    plt.plot(x2, y2, color=color)
    plt.fill_between(x_common, y1_interp, y2_interp, color=color, alpha=0.3)
    return

def plot_roc_ranges(rankings, true_vals, unknown_vals, colors, name_out='out-rocrange.png'):
    plt.figure(figsize=(5,5))
    for r, c in zip(rankings, colors):
        max_x, max_y, min_x, min_y = get_minmax_roc(r, true_vals, unknown_vals)
        plot_range_roc(max_x, max_y, min_x, min_y, color=c)
    plt.xlim(-0.02,1)
    plt.ylim(0,1.02)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.savefig(name_out, dpi=300, bbox_inches='tight')
    plt.show()

def plot_auc_ranges(predictors, pred_vals, true_vals, unknown_vals, name_out='out-auc-range.png'):
    '''
    calcuate best and worst AUC scenarios for each predictor and plot as bar plots
    '''
    # get best case AUC, worst case AUC and current AUC
    current_aucs = []
    bestcase_aucs = []
    worstcase_aucs = []
    for vals in pred_vals:
        x,y = roc_normal(true_vals, vals)
        current_aucs.append(auc(x,y))
        bestcase_auc, worstcase_auc = get_minmax_roc(vals, true_vals, unknown_vals, return_auc=True)
        bestcase_aucs.append(bestcase_auc)
        worstcase_aucs.append(worstcase_auc)
    current_aucs = np.array(current_aucs)
    bestcase_aucs = np.array(bestcase_aucs)
    worstcase_aucs = np.array(worstcase_aucs)

    # plot values
    species = predictors
    values={'ROC':[bestcase_aucs-worstcase_aucs, worstcase_aucs]}
    x = np.arange(len(species))  
    width = 0.55  
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    fig.set_figwidth(5)
    fig.set_figheight(5)
    
    # define colors
    cmap = sns.cubehelix_palette(start=1, rot=-.8, as_cmap=True)
    num_curves = 11
    colors = [cmap(i / (num_curves - 1)) for i in range(num_curves)][::-1]
    
    for attribute, measurement in values.items():
        offset = width * multiplier +0.52
        rects = ax.bar(x + offset, measurement[0], width, label=attribute, bottom=measurement[1], 
                       color=['firebrick']+colors, lw=1, ec='black')
        multiplier += 1
    ax.set_ylabel('ROC AUC')
    print(len(species))
    ax.set_xticks(x + width, species)
    ax.set_ylim(0, 1)
    ax.set_ylim(0.2,0.85)
    plt.xticks(rotation=66)
    plt.hlines(0.5, -0,len(values['ROC'][0]), color='red', alpha=0.5, zorder=-2, linestyle='dashed')
    plt.scatter(np.arange(1,11,1)-0.5, current_aucs, color='black')
    plt.savefig(name_out, dpi=300, bbox_inches='tight')
    plt.show()

def calc_frac_unknown(pred_values, y_true, y_unknown, repeats=1000):
    '''
    for a given selection of unknown mutations, 
    calculate the AUC if X% of unknown mutations are assumed to be stabilizing,
    for a range of X between 0 and 1, with intervals of 0.1. 
    when X% are randomly picked, this is repeated 1000 times for statistical significance. 
    '''
    xp = np.arange(0, 1.05, 0.1)
    auc_preds = []
    # go over each probability
    for probab in np.arange(0, 1.05, 0.1):
        print('generating aucs for frac: ', np.round(probab,1))
        auc_preds_p = []
        # go over each repeat
        for r in range(repeats):
            auc_preds_r = []
            y_randtrue = []
            # generate true/false values for unknown mutations with given probability
            for i, u in zip(y_true, y_unknown):
                if u:
                    y_randtrue.append(np.random.choice([0,1], p=[1-probab, probab]))
                else:
                    y_randtrue.append(i)
            # calculate AUC using new true/false values on given predictors
            for pred in pred_values:
                x,y = roc_normal(y_randtrue, pred)
                auc_preds_r.append(auc(x,y))
            auc_preds_p.append(auc_preds_r)
        auc_preds.append(np.average(np.array(auc_preds_p), axis=0))
    return auc_preds

def plot_auc_ranges_frac(auc_preds, labels, name_out='out-auc-frac.png'):
    '''
    plot how the AUC changes as fraction of mutations in unknown mut
    '''
    # initialize figure
    plt.figure(figsize=((6,6)))
    plt.vlines(0.3, 0,1, color='grey', linestyles='dashed', alpha=0.4)
    plt.hlines(0.5, 0,1, color='grey', linestyles='dashed', alpha=0.4)
    xp = np.arange(0, 1.05, 0.1)

    # define colors
    cmap = sns.cubehelix_palette(start=1, rot=-.8, as_cmap=True)
    num_curves = 11
    colors = [cmap(i / (num_curves - 1)) for i in range(num_curves)] #+ [(), ()]
    order = np.argsort(np.array(auc_preds).T[:,0])
    
    for c, l, y, in zip(colors, np.array(labels)[order], np.array(auc_preds).T[order]):
        if l == 'BoostMut':
            plt.plot(xp, y,  color='firebrick')
            plt.scatter(xp, y, label=l, color='firebrick')
        else:
            plt.plot(xp, y, color=c)
            plt.scatter(xp, y, label=l, color=c)
        n=0.005
        plt.scatter(xp[0]+n, y[0], color='black', zorder=2, lw=2)
    
    plt.xlabel('% stabilizing in untested mutations') 
    plt.ylabel('ROC AUC')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.ylim(0.25, 0.85)
    plt.xticks(xp, (xp*100).astype(int))
    handles, labels = plt.gca().get_legend_handles_labels()
    order = np.arange(0, len(labels))[::-1]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], bbox_to_anchor=(1.1, 1.05))
    plt.savefig(name_out, bbox_inches="tight", dpi=300)
    plt.show()

