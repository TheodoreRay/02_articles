#%% LIBRAIRIES
import traitement_asc as f
import affichage_asc as plot
import matplotlib.pyplot as plt
import numpy as np
from importlib import reload

#%% IMPORTS
s_composant, _ = f.import_data_auto(format='excel')
s_composant = s_composant.set_index('Unnamed: 0', drop=True)

## H1 ##
# résultats de classification #
df_binary, _ = f.import_data_auto(format='excel')
df_binary = df_binary.set_index('date_heure', drop=True)
# description des interventions (inutiles) #
sheet_names = ['X_4}', 'X_0}', 'X_1}', 'X_2}', 'X_3}', 'X_r}', '{md}', 'lon}', 'med}', 'm{MV', 'm{CV']
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
clsfr_shown = ['$\complement_{\epsilon_{X_0}}$','$\complement_{\epsilon_{X_r}}$','$\complement_{\epsilon_{md}}$','$\complement_{\overline{\epsilon}}$', '$\complement_{\epsilon_{med}}$', '$\complement_\mathrm{MV}$', '$\complement_\mathrm{CV}$']

#%% H0
plt.close('all'), reload(plot)
plot.detection_performances(dct_N2I, s_composant, v_turbine, clsfr_shown, '$S_E$')

#%% H1
plt.close('all'), reload(plot)
plot.chronogrammes_classifications(df_binary, dct_2I, clsfr_shown, '$S_E$')
for clsfr in dct_2I.keys():
    IR = int(100*int(dct_2I[clsfr]['durée'].sum()/144)/int(len(df_binary)/144))
    print(f'IR {clsfr} = {IR}')
    ADT = df_binary.index[-1]-dct_2I[clsfr]['début'][0]
    print(f'ADT {clsfr} = {ADT}')