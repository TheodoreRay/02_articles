import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from math import *

## H0 : BARPLOT N_UI ##
def detection_performances(dct_N2I, s_composant, v_turbine, clsfr_shown, wf_name):
        font = {'family' : 'normal', 'weight' : 'normal', 'size' : 24}; matplotlib.rc('font', **font)
        N2I_H0 = dict((clsfr, 0) for clsfr in clsfr_shown)
        #L2I_H0 = dict((clsfr, 0) for clsfr in clsfr_shown)
        plt.figure('N2I H0 parc')
        for x, clsfr in enumerate(clsfr_shown):
                N2I_H0[clsfr] = int(np.sum([dct_N2I[composant][clsfr].sum() for composant in s_composant.index]))
                #L2I_H0[clsfr] = int(np.sum([np.sum([dct_2I[composant][turbine][clsfr]['durée'].sum() for composant in s_composant.index]) for turbine in v_turbine]))
                print(f'N2I (H0) {clsfr} = {N2I_H0[clsfr]}')#"""
                if (clsfr == '$\complement_{\epsilon_{X_0}}$') or (clsfr == '$\complement_{e_{MD}}$') or (clsfr == '$\complement_{\epsilon_{X_r}}$'):
                        plt.bar([.4+.1*x], int(np.ceil(N2I_H0[clsfr]/(len(s_composant)*len(v_turbine)))), tick_label=clsfr, alpha=.7, color='b', edgecolor='k', width=.05)
                else:
                        plt.bar([.4+.1*x], int(np.ceil(N2I_H0[clsfr]//(len(s_composant)*len(v_turbine)))), tick_label=clsfr, alpha=.7, color='r', edgecolor='k', width=.05)
                plt.ylabel('$\overline{N_\mathrm{UI}}$'), plt.grid(axis='y')
                plt.xticks(np.linspace(.4, 1, 7), clsfr_shown)
        plt.yticks(range(0, max([int(np.ceil(N2I_H0[clsfr]/(len(s_composant)*len(v_turbine)))) for clsfr in clsfr_shown])+1, 1))
        #plt.title(wf_name)

## H1 : CHRONOGRAM ##
def chronogrammes_classifications(df_binary, dct_2I, clsfr_shown, wf_name): 
    font = {'family' : 'normal', 'weight' : 'normal', 'size'   : 17}; matplotlib.rc('font', **font)
    plt.subplots(6, 1, sharex=True, sharey=True, num='chronogrammes')
    ax = plt.subplot(7, 1, 1)
    t = [str(t)[:-9] for t in df_binary[clsfr_shown[-1]].index]
  
    for x, clsfr in enumerate(clsfr_shown):
        ## configuration de l'affichage ##
        ax = plt.subplot(7, 1, x+1, sharex=ax, sharey=ax)
        #if x==0: ax.set_title(wf_name)
        ax.plot(df_binary[clsfr], 'k')
        plt.xticks(df_binary[clsfr_shown[-1]].index[::144], [' ']*len(t[::144]))
        plt.yticks([0, 1], [' ', ' ']), plt.ylim(0., 1.), plt.ylabel(clsfr), plt.grid()
        plt.subplots_adjust(wspace = 0, hspace = 0)

        ## coloriage des N2I ##
        # classificateurs mono-résiduels #
        if (clsfr == '$\complement_{\epsilon_{X_0}}$') or (clsfr == '$\complement_{e_{MD}}$') or (clsfr == '$\complement_{\epsilon_{X_r}}$'):
                for d in range(1, len(dct_2I[clsfr[-6: -2]]['début'])):
                        plt.fill_between(df_binary[clsfr].index, df_binary[clsfr], 0, alpha=.5, color='b',\
                                where=(df_binary[clsfr].index>=dct_2I[clsfr[-6: -2]]['début'][d]) & (df_binary[clsfr].index<=dct_2I[clsfr[-6: -2]]['fin'][d]))
                        
                plt.fill_between(df_binary[clsfr].index, df_binary[clsfr], 0, alpha=1.0, color='b',\
                        where=(df_binary[clsfr].index>=dct_2I[clsfr[-6: -2]]['début'][0]) & (df_binary[clsfr].index<=dct_2I[clsfr[-6: -2]]['fin'][0]))
        # classificateurs multi-résiduels #
        else:
                for d in range(1, len(dct_2I[clsfr[-6: -2]]['début'])):
                        plt.fill_between(df_binary[clsfr].index, df_binary[clsfr], 0, alpha=.5, color='r',\
                                where=(df_binary[clsfr].index>=dct_2I[clsfr[-6: -2]]['début'][d]) & (df_binary[clsfr].index<=dct_2I[clsfr[-6: -2]]['fin'][d]))
                plt.fill_between(df_binary[clsfr].index, df_binary[clsfr], 0, alpha=1.0, color='r',\
                        where=(df_binary[clsfr].index>=dct_2I[clsfr[-6: -2]]['début'][0]) & (df_binary[clsfr].index<=dct_2I[clsfr[-6: -2]]['fin'][0]))
    
    ## affichage des dates ##
    xticks = [' ']*len(t[::144])
    xticks[0::2] = t[0::144*2]
    ax.set_xticks(df_binary[clsfr].index[::144], xticks, rotation=35, ha='right')#"""


"""plt.bar(np.arange(.7, len(v_turbine)+.7, 1), dct_N2I[composant].loc[:, '$\complement_{\epsilon_{md}}$'], alpha=.7, \
        tick_label=v_turbine, label='$\complement_{\epsilon_{md}}$', color='tab:pink', edgecolor='k', width=.1)        
plt.bar(np.arange(.8, len(v_turbine)+.8, 1), dct_N2I[composant].loc[:, '$\complement_{\overline{\epsilon}}$'], alpha=.7, \
        tick_label=v_turbine, label='$\complement_{\overline{\epsilon}}$', color='tab:blue', edgecolor='k', width=.1)
plt.bar(np.arange(.9, len(v_turbine)+.9, 1), dct_N2I[composant].loc[:, '$\complement_{\epsilon_{med}}$'], alpha=.7,\
        tick_label=v_turbine, label='$\complement_{\epsilon_{med}}$', color='tab:red', edgecolor='k', width=.1)
plt.bar(np.arange(1., len(v_turbine)+1., 1), dct_N2I[composant].loc[:, '$\complement_{VM}$'], alpha=.7,\
        tick_label=v_turbine, label='$\complement_{VM}$', color='tab:orange', edgecolor='k', width=.1)
plt.bar(np.arange(1.1, len(v_turbine)+1.1, 1), dct_N2I[composant].loc[:, '$\complement_{VA}$'], alpha=.7,\
        tick_label=v_turbine, label='$\complement_{VA}$', color='tab:green', edgecolor='k', width=.1)#"""

"""plt.subplot(212)
plt.bar(np.arange(.6, len(v_turbine)+.6, 1), dct_FPR[composant].loc[:, '$\complement_{\epsilon_{X_0}}$'], alpha=.7, \
        tick_label=v_turbine, label='$\complement_{\epsilon_{X_0}}$', color='tab:purple', edgecolor='k', width=.1)
plt.bar(np.arange(.7, len(v_turbine)+.7, 1), dct_FPR[composant].loc[:, '$\complement_{\epsilon_{md}}$'], alpha=.7, \
        tick_label=v_turbine, label='$\complement_{\epsilon_{md}}$', color='tab:pink', edgecolor='k', width=.1)
plt.bar(np.arange(.8, len(v_turbine)+.8, 1), dct_FPR[composant].loc[:, '$\complement_{\overline{\epsilon}}$'], alpha=.7,\
        tick_label=v_turbine, label='$\complement_{\overline{\epsilon}}$', color='tab:blue', edgecolor='k', width=.1)
plt.bar(np.arange(.9, len(v_turbine)+.9, 1), dct_FPR[composant].loc[:, '$\complement_{\epsilon_{med}}$'], alpha=.7,\
        tick_label=v_turbine, label='$\complement_{\epsilon_{med}}$', color='tab:red', edgecolor='k', width=.1)
plt.bar(np.arange(1., len(v_turbine)+1., 1), dct_FPR[composant].loc[:, '$\complement_{VM}$'], alpha=.7,\
        tick_label=v_turbine, label='$\complement_{VM}$', color='tab:orange', edgecolor='k', width=.1)
plt.bar(np.arange(1.1, len(v_turbine)+1.1, 1), dct_FPR[composant].loc[:, '$\complement_{VA}$'], alpha=.7,\
        tick_label=v_turbine, label='$\complement_{VA}$', color='tab:green', edgecolor='k', width=.1)
plt.ylim(-5, 100), plt.grid(), plt.legend()
plt.xlabel('turbine'), plt.ylabel('FPR(%)')#"""