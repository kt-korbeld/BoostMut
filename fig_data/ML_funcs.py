import numpy as np
import pandas as pd
from sklearn.metrics import auc
from itertools import product

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import sklearn.ensemble
import json
from plotting_funcs import *

def train_model_kfoldcrossval(data_x, data_yc, data_yr, model_names, models, class_reg, weighted=False, rs=42):
    '''
    train a given list of models using K-fold stratified cross validation
    '''
    all_aucs = []
    all_x = {i:[] for i in model_names}
    all_y = {i:[] for i in model_names}
    all_ind_test, all_ind_train = [], []
    rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=10, random_state=rs)
    for train_index, test_index in rskf.split(data_x, data_yc):
        x_train = data_x[train_index]
        x_test = data_x[test_index]
        # get true/false vals for training classifier
        yc_train =  data_yc[train_index]
        yc_test =  data_yc[test_index]
        # get dTms for training regressor
        yr_train =  data_yr[train_index]
        yr_test =  data_yr[test_index]
        # fit model to either classifier or regressor data
        x_preds = []
        for model, cr, in zip(models, class_reg):
            if model == 'mean':
                x_preds.append(x_test.mean(axis=1))
            if cr == 'classifier':
                model.fit(x_train, yc_train)
                x_preds.append(model.predict_proba(x_test)[:,1])
            if cr == 'regressor':
                model.fit(x_train, yr_train)
                x_preds.append(model.predict(x_test))
                
        # use dtm values if weighted, use either dtm or true
        x_out, y_out, aucs_out = [], [], []
        if weighted:
            y_tests = len(models)*[yr_test]
            for y_test, x_pred in zip(y_tests, x_preds):
                x,y = roc_tm(y_test, x_pred)
                x_out.append(x)
                y_out.append(y)
                aucs_out.append(auc(x,y))        
        else:
            #map_test = {'classifier':yc_test, 'regressor':yr_test}
            y_tests = len(models)*[yc_test] #[map_test[i] for i in class_reg]
            
            for y_test, x_pred in zip(y_tests, x_preds):
                x,y = roc_normal(y_test, x_pred)
                x_out.append(x)
                y_out.append(y)
                aucs_out.append(auc(x,y))

        for m, x, y in zip(model_names, x_out, y_out):
            all_x[m]+=[x]
            all_y[m]+=[y]
        all_aucs.append(aucs_out)
    return all_aucs, all_x, all_y

def optimize_params(models, param_grids, model_names, class_reg, X_t, yc_t, yr_t, rs=42):
    '''
    optimize a given set of models for a given set of parameter grids, names, whether it is a classifier/regressor,
    input features (X_t), categorical labels (yc_t) and continious labels (yr_t)
    '''
    best_params_all = {}
    # do manual grid search over given parameter grid
    for model, model_name, cr in zip(models, model_names, class_reg):
        print(model_name)
        if model_name == 'mean':
            continue
        param_grid = param_grids[model_name]
        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]
        best_score = 0
        best_params = []
        for param in param_combinations:
            model.set_params(**param)
            aucs, x, y = train_model_kfoldcrossval(X_t, yc_t, yr_t, [model_name], [model], [cr], weighted=False)
            waucs, x, y = train_model_kfoldcrossval(X_t, yc_t, yr_t, [model_name], [model], [cr], weighted=True)
            # use average of AUC and wAUC as metric for optimizing hyperparameters
            if np.average([aucs, waucs]) > best_score:
                best_score = np.average([aucs, waucs])
                best_params = param
                print(best_score)
        print(model_name, best_params, best_score)
        best_params_all[model_name] = best_params
    return best_params_all

def compare_scenarios(models, model_names, class_reg, data_dict, selections, hyperparams, rs=42):
    '''
    perform k-fold cross validation with different scenarios:
    either with different combinations of proteins, or with or without optimized hyperparameters.
    '''
    aucs_sels_out, waucs_sels_out = [], []
    # include data depending on the specified selections
    for sel in selections:
        X, yc, yr = [], [], [] 
        for d in data_dict.keys():
            if d in sel:
                X += [data_dict[d][0]]
                yc += [data_dict[d][1]]
                yr += [data_dict[d][2]]  
        X = np.vstack(X)
        yc, yr = np.concatenate(yc), np.concatenate(yr)
        yr[yr == '-'] = 0
        yr = yr.astype(float)
        yr[np.isnan(yr)] = 0
        print(X.shape, yc.shape, yr.shape)
        # if 'opt' in selection, use optimized hyperparameters
        if 'opt' in sel:
            print('using optimized hyperparameters')
            for n, m in zip(model_names, models):
                if n == 'mean':
                    continue
                m.set_params(**hyperparams[n])
        all_aucs, all_x, all_y = train_model_kfoldcrossval(X, yc, yr, model_names, models, class_reg, weighted=False)
        all_waucs, all_x, all_y = train_model_kfoldcrossval(X, yc, yr, model_names, models, class_reg, weighted=True)
        aucs_sels_out.append(np.average(all_aucs, axis=0))
        waucs_sels_out.append(np.average(all_waucs, axis=0))
    return aucs_sels_out, waucs_sels_out

def plot_aucs_ml(predictors, auc_vals, wauc_vals, name_out='fig-aucs-ml.png'):
    '''
    plot barplot of AUCs of desired scenario
    '''
    labels = ['ROC', 'Weighted ROC']
    values = {p:[v1, v2] for p, v1, v2 in zip(predictors, auc_vals, wauc_vals)}

    cmap = sns.color_palette("rocket", as_cmap=True)
    num_curves = 8
    colors = ['tab:blue']+[cmap(i / (num_curves - 1)) for i in range(num_curves)][1:] 
    colors = [colors[0], colors[2], colors[4], colors[6]]
    x = np.arange(len(labels))*2  
    width = 0.40  
    multiplier = 0
    
    fig, ax = plt.subplots(layout='constrained', figsize=(7, 5))
    
    
    colors_bars = {n:c for n, c in zip(predictors, colors)}
    add_val = {'ROC':0, 'Weighted ROC':1}
    
    for attribute, measurement in values.items():
        offset = width * multiplier +0.12
        m = tuple([np.round(i, 2) for i in measurement])
        print(attribute, measurement)
        rects = ax.bar(x + offset, m, width, lw=1, ec='black', color=colors[multiplier], label=attribute)
        ax.bar_label(rects, padding=3, fontsize=14)
        multiplier += 1
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('AUC', fontsize=14)
    ax.set_xticks(x + width+0.33, labels, fontsize=14)
    ax.legend(bbox_to_anchor=(1., 1.0), fontsize=14)
    #fig.set_figwidth(7)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.2,3.8)
    plt.savefig(name_out, dpi=300, bbox_inches='tight')
    plt.show()


def plot_scenarios(aucs_sels_out, waucs_sels_out, model_names, scenarios=['separate', 'combined', 'optimized'], combine=3, name_out='fig-scen.png'):
    '''
    plot the delta auc and delta weighted auc compared to the mean for different scenarios as a lineplot
    to show increase in  dAUC as different scenarios are picked. 
    '''

    # round and take difference with mean
    aucs_sels = np.round(np.array(aucs_sels_out), 2)
    auc_sels_diff = aucs_sels[:,1:] - aucs_sels[:,0][:, np.newaxis]
    waucs_sels = np.round(np.array(waucs_sels_out), 2)
    wauc_sels_diff = waucs_sels[:,1:] - waucs_sels[:,0][:, np.newaxis]
    
    # combine the first n entries 
    auc_sels_diff = np.vstack((np.average(auc_sels_diff[:combine,:], axis=0), auc_sels_diff[combine:,:]))
    wauc_sels_diff = np.vstack((np.average(wauc_sels_diff[:combine,:], axis=0), wauc_sels_diff[combine:,:]))

    # plot data
    plt.figure(figsize=(5,5))
    cmap = sns.color_palette("rocket", as_cmap=True)
    num_curves = 8
    colors = ['tab:blue']+[cmap(i / (num_curves - 1)) for i in range(num_curves)][1:]
    colors = [colors[2], colors[4], colors[6]]
    
    for n, i, c in zip(model_names[1:], auc_sels_diff.T, colors):
        plt.plot(scenarios, i, color=c, label=n)
        plt.scatter(scenarios, i, color=c)
    
    for n, i, c in zip(model_names[1:], wauc_sels_diff.T, colors):
        plt.plot(scenarios, i, color=c, linestyle='dashed')
        plt.scatter(scenarios, i, color=c, marker='x')
    
    plt.hlines(0, -1, 4, linestyles='dashed', color='grey', alpha=0.5)
    plt.xlim(-0.1, 2.1)
    plt.ylabel('ΔAUC')
    plt.legend(loc='lower right')
    plt.scatter([-1], [0], label='ROC', color='black')
    plt.scatter([-1], [0], marker='x', label='weighted ROC', color='black')
    legend2 = plt.legend(fontsize=12)
    plt.gca().add_artist(legend2)
    plt.savefig(name_out, dpi=300, bbox_inches='tight')
    plt.show()
