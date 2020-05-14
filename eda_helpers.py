
# script of simple functions to do 
# exploratory data analysis
# will keep adding functions to this
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
from scipy.cluster import hierarchy

def detailed_data_info(df):
    print("column name \t type of data \t\tunique values\t\tWhat data looks like")
    print("------------ \t ----------- \t\t--------------\t\t-------------------")
    for col in df.columns:
        if len(col) < 12:
            col_displayname = col + ' '*(12-len(col))
        else:
            col_displayname = col[:12]

        print("{}  \t {}  \t\t{}\t\t {}".format(col_displayname, 
                df[col].dtype, len(df[col].unique()), df[col].unique()[:3]))


def eda_plot_data(df, plot_type, target=None, max_cols=3, n_plots=9):
    
    n_rows = int(np.ceil(n_plots/max_cols))

    if plot_type == 'dist':
        data_cols = df.select_dtypes(include='number').columns

    elif plot_type == 'count':
        data_cols = df.select_dtypes(exclude='number').columns

    elif plot_type == 'scatter' and target != None:
        data_cols = df.select_dtypes(include='number').columns
    else:
        return "Please specify a target column"

    num_variables = len(data_cols)

    plot_number = 1
    for x in range(0, num_variables, n_plots):
        # initialize subplots
        fig, axs = plt.subplots(n_rows, max_cols, \
            figsize=(12,12))
        axs = axs.ravel()  

        cols = data_cols[x:x+n_plots]
        for i, col in enumerate(cols):
            if plot_type == 'dist':
                # try/except because sometimes the 
                # kernal density estimation fails
                # in that case plot histogram
                try:
                    sns.distplot(df[col], ax=axs[i])
                except RuntimeError:
                    sns.distplot(df[col], kde=False, ax=axs[i])

            elif plot_type == 'count':
                # this can get messy with high cardinality
                unique_categories = df[col].unique()
                if len(unique_categories) <= 20:
                    sns.countplot(df[col], ax=axs[i])
                else:
                    cats_to_plot = np.random.choice(unique_categories, 20)
                    sns.countplot(df[col].loc[df[col].isin(cats_to_plot)], 
                                ax=axs[i])
                    axs[i].set_title(
                        "feat. cardinality {}, random 20 plotted".format(
                            len(unique_categories)))
                axs[i].tick_params(labelrotation=90)
            elif plot_type == 'scatter':
                sns.scatterplot(x=df[col], y=df[target], ax=axs[i], alpha=0.5)

        # format the     
        plt.tight_layout()    
        fig.suptitle("{} plot number {}".format(plot_type, plot_number))
        plt.subplots_adjust(top=0.9)
        plot_number += 1


def plot_correlations(df, target=None):
    # get correlations with pandas
    corr = df.corr()

    # cluster the values and flip for desending order
    order = np.array(hierarchy.dendrogram(
        hierarchy.ward(np.abs(corr)),no_plot=True)['ivl'], dtype="int")

    order = np.flip(order)

    # plot heatmap
    f, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(corr[corr.columns[order]].iloc[order], 
    cmap=sns.diverging_palette(220, 20, as_cmap=True))

    # if the target is specified we can plot the correlation 
    # with each feature
    if target:
        f, ax = plt.subplots(figsize=(9, 6))
        ax.plot(corr.loc[target].abs().sort_values(), '*')
        ax.hlines(y=0.5, xmin=0, xmax=len(corr), linestyles='dashed')
        ax.set_ylabel("correlation with {}".format(target))
        ax.annotate('good correlation', xy=(2, 0.6))
        ax.annotate('weak correlation', xy=(2, 0.4))
        plt.xticks(rotation=90)


