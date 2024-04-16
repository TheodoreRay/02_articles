#%% Librairies
import traitement as f
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from importlib import reload
pd.options.mode.chained_assignment = None

#%% données contextuelles
plt.close('all')
graph = 'mono' # type de visuel
font = {'family' : 'normal', 'weight' : 'normal', 'size'   : 20}; matplotlib.rc('font', **font)
H1 = [['2010-01-26', '2010-02-28'], ['2018-11-30', '2019-09-07'], \
    ['2020-07-01', '2020-08-12'], ['2022-02-15', '2022-08-01']]
turbine = ['T1', 'T1', 'T2', 'T3']
composant = ['roulement 2 génératrice', 'palier arbre lent', 'cooling convertisseur', 'refroidissement gearbox']
légende_mono = [('$\mathrm{E}_f$ (1.6%, 78.1%)', '$\mathrm{E}_f$ (1.6%, 31 days)'), \
    ('$\mathrm{M}_f$ (0.4%, 78.5%)', '$\mathrm{M}_f$ (0.4%, 220 days)'), \
        ('$\mathrm{N}_f$ (3.1%, 81.7%)', '$\mathrm{N}_f$ (3.1%, 42 days)'), \
            ('$\mathrm{V}_f$ (1.7%, 79.4%)', '$\mathrm{V}_f$ (1.7%, 166 days)')]
légende_multi = [['$\mathrm{E}_f$', '$\mathrm{E}_1$'], ['$\mathrm{M}_f$', '$\mathrm{M}_1$', '$\mathrm{M}_2$', '$\mathrm{M}_3$', '$\mathrm{M}_4$', '$\mathrm{M}_5$', '$\mathrm{M}_6$'], \
    ['$\mathrm{N}_f$', '$\mathrm{N}_1$'], ['$\mathrm{V}_f$', '$\mathrm{V}_1$', '$\mathrm{V}_2$', '$\mathrm{V}_3$', '$\mathrm{V}_4$', '$\mathrm{V}_5$', '$\mathrm{V}_6$']]

#%% import des données de la courbe ROC
df_ROC, filename = f.import_data_auto(format='parquet')

#%% CONFIGURATION
i=2 # identifiant de la technologie de turbine
j=0 # identifiant parc

# AFFICHAGE
# code couleur selon le type de visuel
if graph == 'mono':
    markers = ['o', 'v', 'p', 's']
    linestyles = ['dotted', 'dashed', 'dashdot', (0, (5, 10))]
if graph == 'multi':
    color = ['tab:blue','tab:orange','tab:green','tab:purple','tab:brown','tab:pink']

plt.figure('ROC')

# TPR(FPR) #
plt.subplot(121)
if graph == 'mono':
    plt.plot(df_ROC['false positive rate (FPR)'], df_ROC['true positive rate (TPR)'], f"{markers[i]}-", linestyle=linestyles[i], linewidth=2, label = légende_mono[i][0], color="r") 
if graph == 'multi':
    if 'f' in filename:
        plt.plot(df_ROC['false positive rate (FPR)'], df_ROC['true positive rate (TPR)'], ".-", linewidth=2, label = légende_multi[i][j], color='r') 
    else:
        plt.plot(df_ROC['false positive rate (FPR)'], df_ROC['true positive rate (TPR)'], ".-", linewidth=1, alpha=.6, label = légende_multi[i][j], color=color[j-1]) 
    plt.vlines(5, 0, 100, 'r', linewidth=2, linestyle='dotted')
plt.yticks(ticks = np.round(np.arange(0, 110, 10), 1), labels = np.round(np.arange(0, 110, 10), 1), rotation = 0, ha = 'right')
plt.title('TPR(FPR)'), plt.xlabel('FPR(%)'), plt.ylabel('TPR(%)'), plt.grid(visible=True), plt.legend(loc='lower right')

# DA(FPR) #
ymax = np.max(df_ROC['detection advance (DA)'])
plt.subplot(122)
if graph == 'mono':
    plt.plot(df_ROC['false positive rate (FPR)'], df_ROC['detection advance (DA)'], f"{markers[i]}-", linestyle=linestyles[i], linewidth=2, label = légende_mono[i][1], color="r")
if graph == 'multi':
    if 'f' in filename:
        plt.plot(df_ROC['false positive rate (FPR)'], df_ROC['detection advance (DA)'], ".-", linewidth=2, label = légende_multi[i][j], color='r')
    else:
        plt.plot(df_ROC['false positive rate (FPR)'], df_ROC['detection advance (DA)'], ".-", linewidth=1, alpha=.6, label = légende_multi[i][j], color=color[j-1])
    plt.vlines(5, 0, ymax+ymax//10, 'r', linewidth=2, linestyle='dotted')
plt.ylim(-2, ymax+ymax//10)
plt.yticks(ticks = np.arange(0, ymax+ymax//10, ymax//10, dtype=int), labels = np.arange(0, ymax+ymax//10, ymax//10, dtype=int), rotation = 0, ha = 'right')
plt.title('ADT(FPR)'), plt.xlabel('FPR(%)'), plt.ylabel('ADT(days)'), plt.grid(visible=True), plt.legend(loc='lower right') 
plt.xticks(ticks = np.round(np.arange(0, 110, 10), 1), labels = np.round(np.arange(0, 110, 10), 1), rotation = 0, ha = 'right')

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