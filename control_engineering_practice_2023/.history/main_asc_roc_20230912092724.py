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

#%%
sheets = ['epsi_X0', 'epsi_X1', 'epsi_X2', 'epsi_X3', 'epsi_X4', 'epsi_Xr', 'epsi_md', 'overline_epsi', 'epsi_med']
legend = ['$\varepsilon_{X0}$', '$\varepsilon_{X1}$', '$\varepsilon_{X2}$', '$\varepsilon_{X3}$', '$\varepsilon_{X4}$', '$\varepsilon_{Xr}$', '$\varepsilon_{md}$', '$\overline{\varepsilon}$', '$\varepsilon_{med}$']
color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','k']
dct_ROC, _ = f.import_data_auto(format='excel', sheet_names=sheets)
#%%
# ADT(NUI) #
for sheet in sheets:
    plt.plot(dct_ROC[sheet]['NUI'], dct_ROC[sheet]['ADT'], '.', linestyle='solid', label=sheet) 

    #ax[0].vlines(5, 0, 100, 'r', linewidth=2)
    #plt.set_yticks(ticks = np.round(np.arange(0, 110, 5), 1), labels = np.round(np.arange(0, 110, 5), 1), rotation = 0, ha = 'right')
    #plt.set_xticks(ticks = np.round(np.arange(0, 110, 5), 1), labels = np.round(np.arange(0, 110, 5), 1), rotation = 0, ha = 'right')
    plt.set_title('ADT(NUI)'), plt.set_xlabel('NUI'), plt.set_ylabel('ADT (days)'), plt.grid(visible=True), plt.legend(loc='lower right')

#%% CALCUL AUC (trapezoidal rule)
xmin = df_ROC['false positive rate (FPR)'].iloc[-1]
xmax = df_ROC['false positive rate (FPR)'][0]
n = np.shape(df_ROC)[0]
h = (xmax - xmin) / (n - 1)
x = df_ROC['false positive rate (FPR)']
y = df_ROC['true positive rate (TPR)']
#I_riem = np.sum(y[1:]*[x[i-1]-x[i] for i in range(1, n)])/10000
AUC_tpr_fpr = np.sum([(x[i-1]-x[i])*(y[i-1]+y[i]) for i in range(1, n)])/20000
df_AUC.loc['AUC TPR(FPR)', df_valeurs.columns[i]] = AUC_tpr_fpr
print(f'AUC TPR(FPR) = {AUC_tpr_fpr}')
y = df_ROC['detection advance (DA)']
AUC_adt_fpr = np.sum([(x[i-1]-x[i])*(y[i-1]+y[i]) for i in range(1, n)])/(2*100*221)
df_AUC.loc['AUC ADT(FPR)', df_valeurs.columns[i]] = AUC_adt_fpr
print(f'AUC ADT(FPR) = {AUC_adt_fpr}')

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
# %%
