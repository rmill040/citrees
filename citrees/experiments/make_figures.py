import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_friedman1
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from statsmodels.formula.api import ols
import sys
import time

sys.path.append('../')
plt.style.use('ggplot')

from citrees import CIForestClassifier, CIForestRegressor


def tree_splits(tree, splits):
    """Recursively get feature splits from conditional inference model.
    
    Parameters
    ----------
    tree : fitted estimator
        Fitted tree model

    splits : list
        Index of features used for splitting
    
    Returns
    -------
    None
    """
    if tree.value is None:
        splits.append(tree.col)
        tree_splits(tree.left_child, splits)
        tree_splits(tree.right_child, splits)

 
def ensemble_splits(ensemble, sklearn=False):
    """Gather feature splits from fitted ensemble tree model.
    
    Parameters
    ----------
    ensemble : fitted estimator
        Fitted ensemble tree model

    sklearn : bool
        Whether fitted estimator is a sklearn estimator
    
    Returns
    -------
    splits : 1d array-like
        Array of split points
    """
    splits = []
    for estimator in ensemble.estimators_:
        if sklearn:
            tmp = estimator.tree_.feature
            splits.append(tmp[tmp != -2])
        else:
            tmp = []
            tree_splits(estimator.root, tmp) 
            splits.append(tmp)

    return np.concatenate(splits)


def parse_method(name):
    """Parse hyperparameters from string name to make legend label.

    Parameters
    ----------
    name : str
        Name of method

    Returns
    -------
    string : str
        Formatted string
    """
    string = r""
    if name.split('es_')[1][0] == '1':
        string += r'ES'
    if name.split('vm_')[1][0] == '1':
        if len(string) > 0: 
            string += r', VM'
        else:
            string += r'VM'
    alpha = name.split('alpha_')[1].split('_')[0]
    if len(string) > 0: 
        string += r', $\alpha=%s$' % alpha
    else:
        string += r'$\alpha=%s$' % alpha
    return string


def plot_single_df():
    """Plots single dataframe results for classification model.
    """
    # Classification
    df_clf = pd.read_csv('classification/data/classifier_cv_metrics.csv')
    mask   = (df_clf['data'] == 'CLL_SUB_111') & (df_clf['method'].str.startswith('cf'))
    df_clf = df_clf[mask].reset_index(drop=True)
    x      = np.arange(5, 105, 5)
    colors = [
        'grey',
        'tomato',
        'slateblue',
        'darkseagreen',
        'darkslategrey',
        'mediumpurple',
        'royalblue',
        'lightsalmon',
        'plum',
        'indianred',
        'darkolivegreen',
        'rosybrown'
    ]
    for i, method in enumerate(df_clf['method'].unique()):
        label = parse_method(method)
        plt.plot(x, df_clf[df_clf['method'] == method]['auc'], label=label, color=colors[i])
        plt.xticks(x)
    plt.legend()
    plt.xlabel("Number of Features")
    plt.ylabel("AUC")
    plt.xlim([4, 101])
    plt.show()


def plot_split_selection():
    """Plots split selection for conditional inference models and random 
    forest models.
    """
    # Parameters for data size
    data_params = {
        'n_samples'    : 200,
        'n_features'   : 105,
        'noise'        : 5.0,
        'random_state' : None
    }

    # Parameters for conditional forests
    cf_params = {
        'n_estimators'   : 200,
        'max_feats'      : 'sqrt',
        'selector'       : 'pearson',
        'n_permutations' : 150,
        'early_stopping' : True, 
        'muting'         : True,
        'alpha'          : .01,
        'n_jobs'         : -1,
        'verbose'        : False,
        'random_state'   : None
    }

    # Parameters for random forests
    rf_params = {
        'n_estimators' : 200,
        'max_features' : 'sqrt',
        'n_jobs'       : -1,
        'verbose'      : False,
        'random_state' : None
    }

    # Run analysis
    cf_reg_results = []
    rf_reg_results = []
    cf_clf_results = []
    rf_clf_results = []
    for i in range(1, 11):

        print("[info] running iteration %d/10" % i)
        
        # Update random states and generate data
        cf_params['random_state']   = i
        rf_params['random_state']   = i
        data_params['random_state'] = i
        X, y                        = make_friedman1(**data_params)

        # Regression: conditional inference forest
        cf_params['selector'] = 'pearson'
        reg                   = CIForestRegressor(**cf_params).fit(X, y)
        cf_reg_results.append(ensemble_splits(reg))

        # Regression: Random forest
        reg = RandomForestRegressor(**rf_params).fit(X, y)
        rf_reg_results.append(ensemble_splits(reg, sklearn=True))

        # Update data for classification (median split to binarize)
        y = np.where(y > np.median(y), 1, 0)

        # Classification: conditional inference forest
        cf_params['selector'] = 'mc'
        clf                   = CIForestClassifier(**cf_params).fit(X, y)
        cf_clf_results.append(ensemble_splits(clf))

        # Classification: Random forest
        clf = RandomForestClassifier(**rf_params).fit(X, y)
        rf_clf_results.append(ensemble_splits(clf, sklearn=True))

    # Concat results
    cf_reg_results = np.concatenate(cf_reg_results).astype(int)
    rf_reg_results = np.concatenate(rf_reg_results).astype(int)

    cf_clf_results = np.concatenate(cf_clf_results).astype(int)
    rf_clf_results = np.concatenate(rf_clf_results).astype(int)

    # Plot
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

    # [0,Conditional Forest : Regression
    x = np.arange(data_params['n_features'])
    axes[0, 0].bar(x, 100*np.bincount(cf_reg_results)/len(cf_reg_results), color='green')
    axes[0, 0].set_ylabel("Percent")

    # Random Forest : Regression
    axes[0, 1].bar(x, 100*np.bincount(rf_reg_results)/len(rf_reg_results), color="blue")

    # Conditional Forest : Classification
    axes[1, 0].bar(x, 100*np.bincount(cf_clf_results)/len(cf_clf_results), color='green')
    axes[1, 0].set_xlabel("Feature ID")
    axes[1, 0].set_ylabel("Percent")

    # Random Forest : Classification
    axes[1, 1].bar(x, 100*np.bincount(rf_clf_results)/len(rf_clf_results), color="blue")
    axes[1, 1].set_xlabel("Feature ID")

    pad  = 7.5 # in points
    cols = ["Conditional Forest", "Random Forest"]
    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    rows = ["Regression", "Classification"]
    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
    fig.tight_layout()
    plt.show()


def compare_hps():
    """Compares hyperparameters via plots and ANOVAs for classification and regression
    conditional inferernce models.
    """
    # Interactions between hyperparameters in conditional inference clf models
    df    = pd.read_csv('classification/data/classifier_cv_metrics.csv')
    names = df.groupby('data')['n_feats'].min() == 5
    names = names[names == True].index.tolist()
    df_cf = df[(df['method'].str.startswith('cf')) & (df['data'].isin(names))]

    # Get all data sets with high number of features
    mapper                   = {'0': 'False', '1': 'True'}
    method                   = df_cf['method'].apply(lambda x: x.split('_'))
    df_cf['Early Stopping']  = method.apply(lambda x: x[2]).map(mapper)
    df_cf['Variable Muting'] = method.apply(lambda x: x[4]).map(mapper)
    df_cf['Alpha']           = method.apply(lambda x: x[6])

    df_cf = df_cf.rename(
        {'n_feats': 'N Features', 'auc': 'AUC'},
        axis='columns'
        )

    # Average over data to get average auc metrics
    keys  = ['N Features', 'Early Stopping', 'Variable Muting', 'Alpha']
    df_gb = df_cf.groupby(keys)['AUC']\
                .mean()\
                .reset_index()

    # Make facet grid plot
    grid = sns.FacetGrid(
        df_gb, 
        row="Early Stopping", 
        col="Variable Muting", 
        hue="Alpha", 
        )
    grid.map(plt.plot, "N Features", "AUC")

    labels = [r'$\alpha=0.01$', r'$\alpha=0.05$', r'$\alpha=0.95$']
    plt.legend(labels=labels)
    plt.tight_layout()
    plt.show()
    plt.close()

    # Run analysis of variance model
    df_cf.columns = [c.replace(' ', '') for c in df_cf.columns]
    model_str     = 'AUC ~ C(EarlyStopping) + ' \
                          'C(VariableMuting) + ' \
                          'C(Alpha) + ' \
                          'C(EarlyStopping)*C(VariableMuting) + ' \
                          'C(EarlyStopping)*C(Alpha) + ' \
                          'C(VariableMuting)*C(Alpha) + ' \
                          'C(VariableMuting)*C(Alpha)*C(EarlyStopping)'
      
    print(
        "[info] ANOVA - full model classification", 
        ols(model_str, data=df_cf).fit().summary()
    )

    # Interactions between hyperparameters in conditional inference reg models
    df    = pd.read_csv('regression/data/regression_cv_metrics.csv')
    names = df.groupby('data')['n_feats'].min() == 5
    names = names[names == True].index.tolist()
    df_cf = df[(df['method'].str.startswith('cf')) & (df['data'].isin(names))]

    # Get all data sets with high number of features
    method                     = df_cf['method'].apply(lambda x: x.split('_'))
    df_cf['Early Stopping']    = method.apply(lambda x: x[2]).map(mapper)
    df_cf['Variable Muting']   = method.apply(lambda x: x[4]).map(mapper)
    df_cf['Alpha']             = method.apply(lambda x: x[6])
    df_cf['Variable Selector'] = method.apply(lambda x: x[8])

    df_cf = df_cf.rename(
        {'n_feats': 'N Features', 'r2': r'$R^2$'},
        axis='columns'
        )

    # Average over data to get average auc metrics
    keys  = ['N Features', 'Early Stopping', 'Variable Muting', 'Alpha']
    df_gb = df_cf.groupby(keys)[r'$R^2$']\
                .mean()\
                .reset_index()

    # Make facet grid plot
    grid = sns.FacetGrid(
        df_gb, 
        row="Early Stopping", 
        col="Variable Muting", 
        hue="Alpha", 
        )
    grid.map(plt.plot, "N Features", r'$R^2$')
    plt.legend(labels=labels)
    plt.tight_layout()
    plt.show()
    plt.close()

    # Average over data to get average auc metrics
    keys  = ['N Features', 'Early Stopping', 'Variable Muting', 'Variable Selector']
    df_gb = df_cf.groupby(keys)[r'$R^2$']\
                .mean()\
                .reset_index()

    # Make facet grid plot
    grid = sns.FacetGrid(
        df_gb, 
        row="Early Stopping", 
        col="Variable Muting", 
        hue="Variable Selector", 
        )
    grid.map(plt.plot, "N Features", r'$R^2$')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

    # Run analysis of variance model
    df_cf.columns = [c.replace(' ', '') for c in df_cf.columns]
    df_cf         = df_cf.rename({r'$R^2$': 'r2'}, axis='columns')
    model_str     = 'r2 ~ C(EarlyStopping) + ' \
                          'C(VariableMuting) + ' \
                          'C(Alpha) + ' \
                          'C(EarlyStopping)*C(VariableMuting) + ' \
                          'C(EarlyStopping)*C(Alpha) + ' \
                          'C(VariableMuting)*C(Alpha) + ' \
                          'C(VariableMuting)*C(Alpha)*C(EarlyStopping)'
      
    print(
        "[info] ANOVA - full model regression: alpha", 
        ols(model_str, data=df_cf).fit().summary()
    )

    model_str = 'r2 ~ C(EarlyStopping) + ' \
                'C(VariableMuting) + ' \
                'C(VariableSelector) + ' \
                'C(EarlyStopping)*C(VariableMuting) + ' \
                'C(EarlyStopping)*C(VariableSelector) + ' \
                'C(VariableMuting)*C(VariableSelector) + ' \
                'C(VariableMuting)*C(VariableSelector)*C(EarlyStopping)'
      
    print(
        "[info] ANOVA - full model regression: var selector", 
        ols(model_str, data=df_cf).fit().summary()
    )


def test_timings():
    """Test timings for different hp configurations in conditional inference models.
    """
    # Parameters for data size
    data_params = {
        'n_samples'    : 200,
        'n_features'   : None,
        'noise'        : 5.0,
        'random_state' : 1718
    }

    # Parameters for conditional forests
    cf_params = {
        'n_estimators'   : 200,
        'max_feats'      : 'sqrt',
        'n_permutations' : 150,
        'early_stopping' : None, 
        'muting'         : None,
        'alpha'          : None,
        'selector'       : None,
        'n_jobs'         : -1,
        'verbose'        : False,
    }

    df = {
        'type'       : [],
        'es'         : [],
        'vm'         : [],
        'alpha'      : [],
        'time'       : [],
        'iteration'  : [],
        'n_features' : []
     }

    for n_features in range(100, 1050, 50):
        print("n features = %d" % n_features)
        data_params['n_features'] = n_features
        X, y                      = make_friedman1(**data_params)
        y_bin                     = np.where(y > np.median(y), 1, 0)

        for es in [True, False]:
            print("early stopping = %s" % es)
            cf_params['early_stopping'] = es

            for vm in [True, False]:
                print("variable muting = %s" % vm)
                cf_params['muting'] = vm

                for alpha in [.01, .05, .95]:
                    print("alpha = %s" % alpha)
                    cf_params['alpha'] = alpha
                    
                    for i in range(1, 6):
                        print("[info] running iteration %d/5" % i)

                        # Regression: conditional inference forest
                        cf_params['selector'] = 'pearson'
                        start                 = time.time()
                        reg                   = CIForestRegressor(**cf_params).fit(X, y)
                        stop                  = time.time() - start

                        # Update reg data
                        df['type'].append('reg')
                        df['es'].append(es)
                        df['vm'].append(vm)
                        df['alpha'].append(alpha)
                        df['time'].append(stop)
                        df['iteration'].append(i)
                        df['n_features'].append(n_features)

                        # Classification: conditional inference forest
                        cf_params['selector'] = 'mc'
                        start                 = time.time()
                        clf                   = CIForestClassifier(**cf_params).fit(X, y_bin)
                        stop                  = time.time() - start

                        # Update clf data
                        df['type'].append('clf')
                        df['es'].append(es)
                        df['vm'].append(vm)
                        df['alpha'].append(alpha)
                        df['time'].append(stop)
                        df['iteration'].append(i)
                        df['n_features'].append(n_features)

    # Write results to disk
    pd.DataFrame(df).to_csv('timing_results.csv', index=False)


def plot_timings():
    """Plot timing results.
    """
    df = pd.read_csv('timing_results.csv')

    for t in ['clf', 'reg']:
        df_sub = df[df['type'] == t]
        keys   = ['n_features', 'es', 'vm', 'alpha']
        df_gb  = df_sub.groupby(keys)['time']\
                    .apply(lambda x: np.median(x))\
                    .reset_index()

        # Make facet grid plot
        df_gb = df_gb.rename(
            {
            'n_features' : 'N Features',
            'time'       : 'Runtime (seconds)',
            'es'         : 'Early Stopping',
            'vm'         : 'Variable Muting',
            'alpha'      : 'Alpha'
            },
            axis='columns'
        )
        grid = sns.FacetGrid(
            df_gb, 
            row="Early Stopping", 
            col="Variable Muting", 
            hue="Alpha", 
            )
        grid.map(plt.plot, "N Features", 'Runtime (seconds)')
        labels = [r'$\alpha=0.01$', r'$\alpha=0.05$', r'$\alpha=0.95$']
        plt.legend(labels=labels)
        plt.tight_layout()
        plt.show()
        plt.close()

        # Plot times increase in executime time
        df_inc = df_sub.groupby(['es', 'n_features', 'alpha'])['time']\
                    .apply(lambda x: np.median(x))\
                    .reset_index()
        es_true  = df_inc[df_inc['es'] == True] 
        es_false = df_inc[df_inc['es'] == False]
        ratio_01 = es_false[es_false['alpha'] == 0.01]['time'].values / \
                   es_true[es_true['alpha'] == 0.01]['time'].values
        ratio_05 = es_false[es_false['alpha'] == 0.05]['time'].values / \
                   es_true[es_true['alpha'] == 0.05]['time'].values
        ratio_95 = es_false[es_false['alpha'] == 0.95]['time'].values / \
                   es_true[es_true['alpha'] == 0.95]['time'].values
        n_feats = df_inc['n_features'].unique()
        
        plt.plot(n_feats, ratio_01, label=r'$\alpha = .01$')
        plt.plot(n_feats, ratio_05, label=r'$\alpha = .05$')
        plt.plot(n_feats, ratio_95, label=r'$\alpha = .95$')
        plt.xlabel('N Features')
        plt.ylabel('Speedup Factor in Runtime')
        plt.legend(labels=labels)
        plt.tight_layout()
        plt.show()


def main():
    """Main program.
    """
    # Compare results for hyperparameters
    compare_hps()

    # Plot results of conditional inference model for one dataframe
    plot_single_df()
    
    # Run timing experiments and view results
    test_timings()
    plot_timings()

    # Compare split selection with conditional inference model against random forests
    plot_split_selection()

if __name__ == "__main__":
    main()