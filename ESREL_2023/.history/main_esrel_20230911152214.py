#%% Librairies
import traitement_esrel as f
import affichage_esrel as plot
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
import pandas as pd
from importlib import reload
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
pd.options.mode.chained_assignment = None

#%% CONFIGURATION
plt.close('all')
font = {'family' : 'normal', 'weight' : 'normal', 'size' : 15}; matplotlib.rc('font', **font)
dct_H0 = {'SOR':['2010-03-27', '2011-03-27'], 'FOYE':['2020-11-23', '2021-11-23'], 'ABLAI':['2020-08-12', '2021-05-20'], 'PDRS':['2018-02-06', '2018-10-29'], 'MONNE':['2021-02-15', '2022-02-15'], 'LAPLA':['2020-04-01', '2021-04-01'], 'SANTE':['2019-09-20', '2020-09-20']}
dct_H1 = {'SOR':['2010-01-13', '2010-02-28'], 'FOYE':['2021-11-23', '2022-07-01'], 'ABLAI':['2020-06-20', '2020-08-12'], 'PDRS':['2018-11-29', '2019-09-07'], 'MONNE':['2022-02-15', '2022-08-01'], 'LAPLA':['2022-02-02', '2022-02-11'], 'SANTE':['2020-09-20', '2020-12-01']}
#df_vote, _ = f.import_data_auto(format='parquet')
#df_vote.columns=['$\epsilon_{X_0}$', '$\epsilon_{X_1}$', '$\epsilon_{X_2}$', '$\epsilon_{X_3}$', '$\epsilon_{X_4}$', '$\epsilon_{X_r}$', '$\epsilon_{md}$', 'majority vote', '$\overline{\epsilon}$']
#df_vote = df_vote.drop(columns='majority vote')
#df_vote = df_vote.replace(np.nan, 0.5)

#%% IMPORT
WF='FOYE'
df_valeurs, _ = f.import_data_auto(format='parquet')
df_valeurs = df_valeurs.drop(columns='y')
df_valeurs.columns=['$\epsilon_{X_0}$', '$\epsilon_{X_1}$', '$\epsilon_{X_2}$', '$\epsilon_{X_3}$', '$\epsilon_{X_4}$', '$\epsilon_{X_r}$', '$\epsilon_{md}$', '$\overline{\epsilon}$']

#%% LINECHART
plt.close('all')
font = {'family' : 'normal', 'weight' : 'normal', 'size' : 16}; matplotlib.rc('font', **font)
_, ax = plt.subplots(1, 1, num='comparaison résidus - line plot')
if WF=='SOR':
    df_valeurs = pd.concat([df_valeurs[dct_H1[WF][0]:dct_H1[WF][1]], df_valeurs[dct_H0[WF][0]:dct_H0[WF][1]]])
for col in ['$\epsilon_{X_0}$']:#df_valeurs.columns:
    #df_valeurs = df_valeurs.loc[:dct_H1[WF][1], :]
    #df_vote = df_vote.loc[:dct_H1[WF][1], :]
    ## affichage des line plots par résidu ##
    if col=='$\epsilon_{X_0}$':#'$\overline{\epsilon}$':
        ax.plot(df_valeurs[col][::144], 'k', linewidth = 2, label=col)
    elif col=='$\epsilon_{md}$':
        ax.plot(df_valeurs[col][::144], 'grey', linestyle='dotted', label=col)
    else:
        ax.plot(df_valeurs[col][::144], linestyle ='dashed', alpha=.6, label=col)
    ## calcul des performances de détection ##
    #print(f'TFP {col} : {df_vote.loc[dct_H0[WF][0]:dct_H0[WF][1], col].sum()/len(df_vote.loc[dct_H0[WF][0]:dct_H0[WF][1], col].dropna())}')
    #print(f'TFN {col} : {df_vote.loc[dct_H1[WF][0]:dct_H1[WF][1], col].value_counts().loc[0]/len(df_vote.loc[dct_H1[WF][0]:dct_H1[WF][1], col].dropna())}')
    print(f'TI (H0) {col} : {df_valeurs.loc[dct_H0[WF][0]:dct_H0[WF][1], col].dropna().count()/df_valeurs.loc[dct_H0[WF][0]:dct_H0[WF][1], col].count()}')
    print(f'TI (H1) {col} : {df_valeurs.loc[dct_H1[WF][0]:dct_H1[WF][1], col].dropna().count()/df_valeurs.loc[dct_H1[WF][0]:dct_H1[WF][1], col].count()}')
    #if df_vote.loc[dct_H1[WF][0]:dct_H1[WF][1], col].value_counts().loc[0]/len(df_vote.loc[dct_H1[WF][0]:dct_H1[WF][1], col].dropna()) < 1:
    #    print(f'date de détection {col} : {f.AVD_daily(df_vote.loc[dct_H1[WF][0]:dct_H1[WF][1], col])[1]}')

ax.add_patch(Rectangle(xy=(mdates.date2num(np.datetime64(dct_H0[WF][0])), -1), \
    width=mdates.date2num(np.datetime64(dct_H0[WF][1]))-mdates.date2num(np.datetime64(dct_H0[WF][0])), \
    height=2, edgecolor='green', facecolor='palegreen', lw=3, linestyle='dashed', label='$H_0$'))
ax.add_patch(Rectangle(xy=(mdates.date2num(np.datetime64(dct_H1[WF][0])), -1), \
    width=mdates.date2num(np.datetime64(dct_H1[WF][1]))-mdates.date2num(np.datetime64(dct_H1[WF][0])), \
    height=2, edgecolor='red', facecolor='lightcoral', lw=3, linestyle='dashed', label='$H_1$'))
#ax.add_patch(Rectangle(xy=(mdates.date2num(np.datetime64(dct_H0[WF][1])), -1), \
#    width=mdates.date2num(np.datetime64(dct_H1[WF][0]))-mdates.date2num(np.datetime64(dct_H0[WF][1])), \
#    height=2, edgecolor=(.1,.1,.1), facecolor=(.7,.7,.7), lw=3, linestyle='dashed', label='$H_w$'))
ax.set_xticks([str(d)[:-9] for d in df_valeurs.index[::144*25]], [str(d)[:-9] for d in df_valeurs.index[::144*25]], rotation=45, ha='right')
ax.set_yticks(np.arange(-8,20,2), np.arange(-8,20,2))#-8, [str(d)[:-9] for d in df_valeurs.index[::144*25]], rotation=45, ha='right')
ax.grid()#, ax.legend(ncol=2, loc='lower right')

#%% MATRICES D'INTERCORRELATION
plt.close('all')
_, ax = plt.subplots(1, 2, num='correlation matrix')
cmap = 'RdYlGn'
df_H0 = df_valeurs[dct_H0[WF][0]: dct_H0[WF][1]]
mask = np.triu(np.ones_like(df_H0.corr(), dtype=bool))
sns.heatmap(df_H0.corr(), mask=mask, annot=True, cmap=cmap, vmin=0, vmax=1, center=0.5,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax[0])
ax[0].set_xticks(np.arange(len(df_valeurs.columns))+0.5, list(df_valeurs.columns))
df_H1 = df_valeurs[dct_H1[WF][0]: dct_H1[WF][1]]
mask = np.triu(np.ones_like(df_H1.corr(), dtype=bool))
sns.heatmap(df_H1.corr(), mask=mask, annot=True, cmap=cmap, vmin=0, vmax=1, center=0.5,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax[1])
ax[1].set_xticks(np.arange(len(df_valeurs.columns))+0.5, list(df_valeurs.columns))

#%% ROC CURVES
plt.close('all')
_, ax = plt.subplots(1, 2, num='ROC curves')
df_AUC = pd.DataFrame(index=['AUC TPR(FPR)', 'AUC ADT(FPR)'], columns=df_valeurs.columns)

#%%
i = 0 # identifiant du résidu
color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','k']
df_ROC, _ = f.import_data_auto(format='parquet')
id_5p = (np.abs(df_ROC['false positive rate (FPR)']-5)).idxmin() # point de mesure le plus proche de FPR=5%

# TPR(FPR) #
if i==7:
    ax[0].plot(df_ROC['false positive rate (FPR)'], df_ROC['true positive rate (TPR)'], '.-', linewidth=2, color=color[i]) 
    ax[0].plot(df_ROC.loc[id_5p, 'false positive rate (FPR)'], df_ROC.loc[id_5p, 'true positive rate (TPR)'], '.', markersize=20, label=df_valeurs.columns[i], color=color[i]) 
else:
    ax[0].plot(df_ROC['false positive rate (FPR)'], df_ROC['true positive rate (TPR)'], '.', linestyle='dashed', color=color[i]) 
    ax[0].plot(df_ROC.loc[id_5p, 'false positive rate (FPR)'], df_ROC.loc[id_5p, 'true positive rate (TPR)'], '.', markersize=20, fillstyle = 'none', linestyle='dashed', label=df_valeurs.columns[i], color=color[i]) 
#ax[0].vlines(5, 0, 100, 'r', linewidth=2)
ax[0].set_yticks(ticks = np.round(np.arange(0, 110, 5), 1), labels = np.round(np.arange(0, 110, 5), 1), rotation = 0, ha = 'right')
ax[0].set_xticks(ticks = np.round(np.arange(0, 110, 5), 1), labels = np.round(np.arange(0, 110, 5), 1), rotation = 0, ha = 'right')
ax[0].set_title('TPR(FPR)'), ax[0].set_xlabel('FPR(%)'), ax[0].set_ylabel('TPR(%)'), ax[0].grid(visible=True), ax[0].legend(loc='lower right')

# DA(FPR) #
if i==7:
    ax[1].plot(df_ROC['false positive rate (FPR)'], df_ROC['detection advance (DA)'], '.-', linewidth=2, color=color[i])
    ax[1].plot(df_ROC.loc[id_5p, 'false positive rate (FPR)'], df_ROC.loc[id_5p, 'detection advance (DA)'], '.', markersize=20, label=df_valeurs.columns[i], color=color[i]) 
else:
    ax[1].plot(df_ROC['false positive rate (FPR)'], df_ROC['detection advance (DA)'], '.', linestyle='dashed', color=color[i])
    ax[1].plot(df_ROC.loc[id_5p, 'false positive rate (FPR)'], df_ROC.loc[id_5p, 'detection advance (DA)'], '.', markersize=20, fillstyle = 'none', linestyle='dashed', label=df_valeurs.columns[i], color=color[i]) 
#ax[1].vlines(5, 0, df_ROC['detection advance (DA)'].max(), 'r', linewidth=2)
ax[1].set_yticks(ticks = np.arange(0, np.max(df_ROC['detection advance (DA)'])+20, 20, dtype=int), labels = np.arange(0, np.max(df_ROC['detection advance (DA)'])+20, 20, dtype=int), rotation = 0, ha = 'right')
ax[1].set_xticks(ticks = np.round(np.arange(0, 110, 5), 1), labels = np.round(np.arange(0, 110, 5), 1), rotation = 0, ha = 'right')
ax[1].set_title('ADT(FPR)'), ax[1].set_xlabel('FPR(%)'), ax[1].set_ylabel('ADT(days)'), ax[1].grid(visible=True), ax[1].legend(loc='lower right') 

# CALCUL AUC (trapezoidal rule)
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
