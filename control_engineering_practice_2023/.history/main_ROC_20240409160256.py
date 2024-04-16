#%% Librairies
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from importlib import reload
pd.options.mode.chained_assignment = None

#%% CONFIGURATION
font = {'family' : 'normal', 'weight' : 'normal', 'size'   : 25}; matplotlib.rc('font', **font)
parc = ['Efrg2', '$\mathrm{E}_f^{rg2}$'] #id fichier : légende
#parc = ['Mfrm', '$\mathrm{M}_f^{rm}$']
s_indicateur = {'epsiX0':'$\epsilon_{X_0}$', 'epsiX1':'$\epsilon_{X_1}$','epsiX2':'$\epsilon_{X_2}$',\
                'epsiX3':'$\epsilon_{X_3}$', 'epsiX4':'$\epsilon_{X_4}$','epsiXr':'$\epsilon_{X_r}$',\
                'eMD':'$e_\mathrm{MD}$'}#,'epsiMoyen':'$\overline{\epsilon}$','epsiMed':'$\epsilon_\mathrm{Me}$'}

#%% AFFICHAGE
plt.close('all')
color = ['tab:blue','tab:orange','tab:green','tab:purple','tab:brown','tab:pink']

plt.figure('ROC')

for indicateur in s_indicateur.keys():
    df_ROC = pd.read_parquet(f'../../../0_codes/02_articles/control_engineering_practice_2023/df_ROC_{parc[0]}_{indicateur}.parquet')
    n = np.shape(df_ROC['false positive rate (FPR)'])[0]
    # Area Under Curve (AUC) #
    AUC = int(100*np.sum([(np.max(df_ROC['false positive rate (FPR)'].iloc[i-1]-\
                                  df_ROC['false positive rate (FPR)'].iloc[i], 0)*\
                                    (df_ROC['true positive rate (TPR)'].iloc[i]+\
                                     df_ROC['true positive rate (TPR)'].iloc[i-1]))/2 for i in range(1, n)])/(100*100))
    # TPR(FPR) #
    if indicateur == 'epsiX0':
        plt.plot(df_ROC['false positive rate (FPR)'], df_ROC['true positive rate (TPR)'], \
             ".-", linewidth=2, alpha=1, label = f'{s_indicateur[indicateur]}, AUC={AUC}%') 
    else:
        plt.plot(df_ROC['false positive rate (FPR)'], df_ROC['true positive rate (TPR)'], \
             ".-", linewidth=1, alpha=.6, label = f'{s_indicateur[indicateur]}, AUC={AUC}%') 
    #plt.vlines(5, 0, 100, 'r', linewidth=2, linestyle='dotted')
    plt.yticks(ticks = np.round(np.arange(0, 110, 10), 1), \
               labels = np.round(np.arange(0, 110, 10), 1), rotation = 0, ha = 'right')
    plt.title('TVP(TFP)')
    plt.xlabel('TFP(%)')
    plt.ylabel('TVP(%)')
    plt.grid(visible=True)
    plt.legend(loc='lower right')

    # DA(FPR) #
    """ymax = np.max(df_ROC['detection advance (DA)'])
    plt.plot(df_ROC['false positive rate (FPR)'], df_ROC['detection advance (DA)'], ".-", linewidth=1, alpha=.6, label = légende_multi[i][j], color=color[j-1])
    plt.vlines(5, 0, ymax+ymax//10, 'r', linewidth=2, linestyle='dotted')
    plt.ylim(-2, ymax+ymax//10)
    plt.yticks(ticks = np.arange(0, ymax+ymax//10, ymax//10, dtype=int), labels = np.arange(0, ymax+ymax//10, ymax//10, dtype=int), rotation = 0, ha = 'right')
    plt.title('ADT(FPR)'), plt.xlabel('FPR(%)'), plt.ylabel('ADT(days)'), plt.grid(visible=True), plt.legend(loc='lower right') 
    plt.xticks(ticks = np.round(np.arange(0, 110, 10), 1), labels = np.round(np.arange(0, 110, 10), 1), rotation = 0, ha = 'right')#"""

#%% codes sources
#%% BANC ESSAI : étude généralisabilité échelle technologie
"""reload(plot)
modele_turbine = ['V100']*6 #['G90', 'G87', 'G97']
v_parc = ['NEUIL', 'AZERA', 'STSEB', 'STPER', 'PLESI', 'JASSE']#['SMCC', 'SPDB', 'LAPLA', 'FOYE', 'STHIL', 'CHALE']#
v_parc_latex = ['VA', 'VB', 'VC', 'VD', 'VE', 'VF']
for m, parc in enumerate(v_parc):
    # import modèles #
    print(parc)
    dct_models[parc] = pd.read_excel(f"../../../1_data/12_learning/{modele_turbine[m]}_{parc}_S2EV.xlsx", sheet_name=None)
    for key in dct_models[parc]:
        dct_models[parc][key] = dct_models[parc][key].set_index(dct_models[parc][key].iloc[:, 0]).iloc[:, 1:]
    v_composant = pd.Series(index = list(dct_models[parc].keys())[:-3], data = [dct_models[parc][compo].index.name for compo in list(dct_models[parc].keys())[:-3]])
    # update variables modèles #
    dct_models, YX, YX_learning, dct_MAE = model_generation(dct_models, parc, v_composant, [], v_turbine, data_learning, data_full)
    # construction des TSI #
    _, TSI_std, id_nan, _, perfs, _ = TSI_building_and_evaluating(dct_models, parc, v_turbine, [composant], H0, H1[3], YX, YX_learning)
    # comparaison multimodèle des performances #
    df_ROC = plot.ROC(TSI_std[composant][turbine], id_nan[composant][turbine], H0, H1[3], [v_parc_latex[m], v_parc_latex[m]], m+1)
    df_ROC.to_parquet(f'{v_parc_latex[m]}_ROC.parquet')#"""