#%% LIBRAIRIES
import traitement_cep as f
import affichage_cep as plot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
from importlib import reload

#%% IMPORTS - Figure 5 (ROC) (SOR, roulement 2 génératrice, T1) (H0, H1)
df_epsi, _ = f.import_data_auto(format='parquet')

#%% Figure 3 - courbe ROC NUI(ADT)
#%% dct_ROC ##
dct_ROC = dict((label, pd.DataFrame(index = np.arange(0, 3.1, 0.1), columns = ['ADT', 'NUI'])) for label in df_epsi.columns)
#%% dct_ROC - ADT (H1) ##
for label in df_epsi.columns:
    for tau in np.arange(0, 3.1, 0.1):
        print(f'tau={tau}')
        df_epsi_binary = df_epsi[label].dropna().where(df_epsi[label] > tau, 0).where(df_epsi[label]<=tau, 1)
        dates, _ = f.alert(df_epsi_binary, 0.5)
        if len(dates['début'])>0:
            ADT = df_epsi_binary.index[-1] - dates['début'][0]
        else:
            ADT = np.nan
        print(f'ADT={ADT}\n')
        dct_ROC[label].loc[tau, 'ADT'] = ADT

#%% dct_ROC - N2I (H0) ##
for label in df_epsi.columns:
    for tau in np.arange(0, 3.1, 0.1):
        print(f'tau={tau}')
        df_epsi_binary = df_epsi[label].dropna().where(df_epsi[label]>tau, 0).where(df_epsi[label]<=tau, 1)
        _, N2I = f.alert(df_epsi_binary, 0.5)
        print(f'N2I={N2I}')
        dct_ROC[label].loc[tau, 'NUI'] = N2I

#%% affichage courbes ROC ##
plt.close('all')
font = {'family' : 'normal', 'weight' : 'normal', 'size' : 22}; matplotlib.rc('font', **font)
color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','k']
ADT_min = int(str(np.min([dct_ROC[label]['ADT'].min() for label in df_epsi.columns]))[:2])
ADT_max = int(str(np.max([dct_ROC[label]['ADT'].max() for label in df_epsi.columns]))[:2])
NUI_min = np.min([dct_ROC[label]['NUI'].min() for label in df_epsi.columns])
NUI_max = np.max([dct_ROC[label]['NUI'].max() for label in df_epsi.columns])

for c, label in enumerate(df_epsi.columns.difference(['$\overline{\epsilon}$', '$\epsilon_{Me}$'])):
    x = dct_ROC[label]['NUI']
    y = [int(str(dct_ROC[label]['ADT'].iloc[i])[:2]) for i in range(len(dct_ROC[label]))]
    dct_ROC[label] = dct_ROC[label].sort_values(by='NUI', ascending=False)
     
    xmin = int(str(dct_ROC[label]['NUI'].min())[:2])
    xmax = int(str(dct_ROC[label]['NUI'].max())[:2])
    n = np.shape(dct_ROC[label])[0]

    ## AUC (trapezoidal rule) ##
    AUC = int(100*np.sum([(np.max(x.iloc[i-1]-x.iloc[i], 0)*(y[i]+y[i-1]))/2 for i in range(1, n)])/(xmax*ADT_max))

    ## ADT(NUI) ##
    if label in ['$\epsilon_{X_0}$', '$\epsilon_{X_r}$', '$e_{MD}$']:
        plt.plot(x, y, 'o', color=color[c], alpha = 1., linewidth=2.5, linestyle='solid', label=f'{label}, AUC={AUC}%')
    else:   
        plt.plot(x, y, '.', color=color[c], alpha = .8, linewidth=.8, linestyle='dashed', label=f'{label}, AUC={AUC}%')
    plt.yticks(ticks = np.round(np.arange(ADT_min, ADT_max, 2), 0), labels = np.round(np.arange(ADT_min, ADT_max, 2), 0), rotation = 0, ha = 'right')
    plt.xticks(ticks = np.round(np.arange(NUI_min, NUI_max, 2), 0), labels = np.round(np.arange(NUI_min, NUI_max, 2), 0), rotation = 0, ha = 'right')
    plt.xlabel('$N_{UI}$'), plt.xlim(-1, 24), plt.ylabel('$ADT$ (days)')
    plt.grid(visible=False), plt.legend(loc=(0.75, 0.3))#, bbox_to_anchor=(1, 0.1))

#%% IMPORTS - Figure 6, 7 (barplot, chronogramme)
s_composant, _ = f.import_data_auto(format='excel')
s_composant = s_composant.set_index('Unnamed: 0', drop=True)

## H1 ##
# résultats de classification #
df_binary, _ = f.import_data_auto(format='excel')
df_binary = df_binary.set_index('date_heure', drop=True)
# description des interventions (inutiles) #
sheet_names = ['X_4}', 'X_0}', 'X_1}', 'X_2}', 'X_3}', 'X_r}', '{MD}', 'lon}', '{Me}', 'm{MV', 'm{CV']
dct_2I, _ = f.import_data_auto(format='excel', sheet_names=sheet_names)
for key in dct_2I.keys(): dct_2I[key] = dct_2I[key].set_index('Unnamed: 0', drop=True)
clsfr = list(df_binary.columns)

## H0 ##
# nombre d'interventions inutiles #
dct_N2I, _ = f.import_data_auto(format='excel', sheet_names=s_composant.index)
for key in dct_N2I.keys(): dct_N2I[key] = dct_N2I[key].set_index('Unnamed: 0', drop=True)

#%% CONFIGURATION
v_turbine = list(dct_N2I[s_composant.index[0]].index)
composant = 'roulement 2 génératrice'
turbine = 'T1'
clsfr_shown = ['$\complement_{\epsilon_{X_0}}$','$\complement_{\epsilon_{X_r}}$','$\complement_{e_{MD}}$','$\complement_{\overline{\epsilon}}$', '$\complement_{\epsilon_{Me}}$', '$\complement_\mathrm{MV}$', '$\complement_\mathrm{CV}$']

#%% Figure 6 - barplots H0
plt.close('all'), reload(plot)
plot.detection_performances(dct_N2I, s_composant, v_turbine, clsfr_shown, '$S_E$')

#%% Figure 7 - chronogrammes H1
plt.close('all'), reload(plot)
plot.chronogrammes_classifications(df_binary, dct_2I, clsfr_shown, '$S_E$')
for clsfr in dct_2I.keys():
    IR = int(100*int(dct_2I[clsfr]['durée'].sum()/144)/int(len(df_binary)/144))
    print(f'IR {clsfr} = {IR}')
    ADT = df_binary.index[-1]-dct_2I[clsfr]['début'][0]
    print(f'ADT {clsfr} = {ADT}')
#%% Figure 7-bis - chronogrammes H1 7 indicateurs résiduels
plt.close('all'), reload(plot)
clsfr_shown = ['$\complement_{\epsilon_{X_0}}$','$\complement_{\epsilon_{X_1}}$','$\complement_{\epsilon_{X_2}}$','$\complement_{\epsilon_{X_3}}$','$\complement_{\epsilon_{X_4}}$','$\complement_{\epsilon_{X_r}}$','$\complement_{e_{MD}}$']
plot.chronogrammes_classifications(df_binary, dct_2I, clsfr_shown, '$S_E$')
