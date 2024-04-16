#%% Librairies
import matplotlib.pyplot as plt
import matplotlib
import traitement_asc as f
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

#%% CONFIGURATION
plt.close('all')
font = {'family' : 'normal', 'weight' : 'normal', 'size' : 15}; matplotlib.rc('font', **font)

sheets = ['epsi_X0', 'epsi_X1', 'epsi_X2', 'epsi_X3', 'epsi_X4', 'epsi_Xr', 'epsi_md', 'overline_epsi', 'epsi_med']
legend = ['$\epsilon_{X0}$', '$\epsilon_{X1}$', '$\epsilon_{X2}$', '$\epsilon_{X3}$', '$\epsilon_{X4}$', '$\epsilon_{Xr}$', '$\epsilon_{md}$', '$\overline{\epsilon}$', '$\epsilon_{med}$']
color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','k']
dct_ROC, _ = f.import_data_auto(format='excel', sheet_names=sheets)

#%% MAIN
NUI_max = np.max([dct_ROC[sheet]['NUI'].max() for sheet in sheets])
ADT_max = np.max([dct_ROC[sheet]['ADT'].max() for sheet in sheets])
for i, sheet in enumerate(sheets):
    ## ADT(NUI) ##
    if i < 7:
        plt.plot(dct_ROC[sheet]['NUI'], dct_ROC[sheet]['ADT'], '.', linestyle='dashed', label=legend[i]) 
    else:
        plt.plot(dct_ROC[sheet]['NUI'], dct_ROC[sheet]['ADT'], '.', linestyle='solid', label=legend[i]) 
    #ax[0].vlines(5, 0, 100, 'r', linewidth=2)
    #plt.set_yticks(ticks = np.round(np.arange(0, 110, 5), 1), labels = np.round(np.arange(0, 110, 5), 1), rotation = 0, ha = 'right')
    #plt.set_xticks(ticks = np.round(np.arange(0, 110, 5), 1), labels = np.round(np.arange(0, 110, 5), 1), rotation = 0, ha = 'right')
    plt.title('ADT(NUI)'), plt.xlabel('NUI'), plt.ylabel('ADT (days)')
    plt.grid(visible=True), plt.legend(loc='lower right')
    ## AUC (trapezoidal rule) ##
    xmin = dct_ROC[sheet]['NUI'].min()
    xmax = dct_ROC[sheet]['NUI'].max()
    n = np.shape(dct_ROC[sheet])[0]
    h = (xmax - xmin) / (n - 1)
    x = dct_ROC[sheet]['NUI']
    y = dct_ROC[sheet]['ADT']
    AUC = np.sum([((x[i-1]-x[i])*(y[i]+y[i-1]))/2 for i in range(1, n)])/(xmax*ADT_max)
    print(f'AUC {sheet} = {AUC}')
    #print(dct_ROC[sheet]['NUI'].max()*dct_ROC[sheet]['ADT'].max())

## affichage de la heatmap ##
"""_, ax = plt.subplots(1, 1, num='comparaison résidus - heatmap')
df_1 = df_vote.copy()
df_1.iloc[:, -1] = float('nan')
ax = sns.heatmap(df_1.astype(float).T, cmap = 'binary', cbar=False, annot = False, fmt=".0f", annot_kws={'size':6})
df_2 = df_vote.copy()
df_2.iloc[:, :-1] = float('nan')
sns.heatmap(df_2.astype(float).T, cmap = 'Reds', cbar=False, annot = False, fmt=".0f", annot_kws={'size':6})
for i in range(df_vote.shape[1]+1): ax.axhline(i, color='black', lw=2)
ax.set_xticks(range(0, len(df_vote.index), 25), df_vote.index[::25], rotation=45, ha='right')
ax.set_yticks(range(0, len(df_vote.columns)), df_vote.columns, rotation=0, ha='right')
#"""