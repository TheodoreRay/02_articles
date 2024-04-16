import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
import random
import datetime
import missingno as msno
import traitement_esrel as func
import mplcursors
from sklearn.metrics import mean_absolute_error
from math import *
from dateutil.rrule import rrule, DAILY, WEEKLY, MONTHLY, MINUTELY
from matplotlib.patches import Rectangle
from importlib import reload

#%% FONCTIONS
#%% UTILITAIRES ##
def xaxis_config(t, ticks_number):
    if int(len(t)/ticks_number) != 0:
        labels = [str(t.iloc[i])[2:-8] for i in range(0, len(t), int(len(t)/ticks_number))]
        #plt.xticks(ticks = np.linspace(0, len(t), ticks_number+1), labels = labels, rotation = 90, ha = 'right')
        plt.xticks(ticks = t.iloc[::int(len(t)/ticks_number)], labels = labels, rotation = 40, ha = 'right')
    else:
        labels = [str(t.iloc[i])[2:-8] for i in range(0, len(t))]
        #plt.xticks(ticks = np.linspace(0, len(t), ticks_number+1), labels = labels, rotation = 90, ha = 'right')
        plt.xticks(ticks = t.iloc[:], labels = labels, rotation = 40, ha = 'right')

def annotations(y, dates, textes):
    for x in range(len(dates)):
        x_coord = datetime.datetime.strptime(dates[x],"%Y-%m-%d %H:%M:%S")
        plt.annotate(textes[x],
            xy=(x_coord, y[dates[x]]), xycoords='data',
            xytext=(x_coord, y.mean()-1), textcoords='data',
            arrowprops=dict(arrowstyle="fancy", color='black', connectionstyle="arc3"), 
            bbox=dict(boxstyle="round", fc="0.8"),
            ha="center", va="center"
            )

#%% ANALYSE DES DONNEES SCADA ##
def variables_modele(v_turbine, data, data_models, composant, mois, annee, f):
    font = {'family' : 'normal', 'weight' : 'normal', 'size'   : 10}; matplotlib.rc('font', **font)
    v_variable = [data_models[composant].index.name] + list(data_models[composant].index)[:3]
    variables(v_turbine, data, v_variable, mois, annee, f)

def variables(v_turbine, data, v_variable, debut, fin):
    font = {'family' : 'normal', 'weight' : 'normal', 'size'   : 10}; matplotlib.rc('font', **font)
    _, axs = plt.subplots(len(v_variable), 1, sharey = True, sharex = True)
    data = data[(data['date_heure']>debut) & (data['date_heure']<fin)]
    ticks_number = 30
    ## EXTRACTION DES SIGNAUX ##
    for var, variable in enumerate(v_variable):
        for turbine in v_turbine:
            t = data[data['ref_turbine_valorem'] == turbine].fillna(0)['date_heure']
            y = data[data['ref_turbine_valorem'] == turbine][variable].fillna(0).ewm(alpha=0.01).mean()
            if len(v_variable)>1:
                axs[var].plot(t, y, '.-', label = turbine)
            else:
                plt.plot(t, y, '.-', label = turbine)
        # configuration des plots #
        if len(v_variable)>1:
            axs[var].set_title(variable), xaxis_config(t, ticks_number), axs[var].set_xticklabels([]), axs[var].grid()
        else:
            plt.title(variable), plt.grid()
    plt.legend(loc = 'lower right'), xaxis_config(t, ticks_number)

def turbines(v_turbine, data, v_variable, debut, fin):
    font = {'family' : 'normal', 'weight' : 'normal', 'size'   : 10}; matplotlib.rc('font', **font)
    _, axs = plt.subplots(len(v_turbine), 1, sharey = True, sharex = True)
    data = data[(data['date_heure']>debut) & (data['date_heure']<fin)]
    ticks_number = 30
    ## EXTRACTION DES SIGNAUX ##
    for tur, turbine in enumerate(v_turbine):
        for variable in v_variable:
            t = data[data['ref_turbine_valorem'] == turbine].fillna(0)['date_heure']
            y = data[data['ref_turbine_valorem'] == turbine][variable].fillna(0).ewm(alpha=0.01).mean()
            if len(v_turbine)>1:
                axs[tur].plot(t, y, label = variable)
            else:
                plt.plot(t, y, label = variable)
        # configuration des plots #
        if len(v_turbine)>1:
            axs[tur].set_title(turbine), xaxis_config(t, ticks_number), axs[tur].set_xticklabels([])
        else:
            plt.title(turbine), plt.grid()
    plt.legend(loc = 'lower right'), xaxis_config(t, ticks_number)

def amplitude_fft(data, variable, turbine, do_plot):
    y = data[data['ref_turbine_valorem'] == turbine][variable].dropna().values
    N = len(y)
    # calcul fft/amplitude #
    yf = np.fft.fft(y)
    freq = np.fft.fftfreq(np.arange(len(yf)).shape[-1])
    amp_fft = 2.0/N * np.median(np.abs(yf[0:N//2]))
    # affichage #
    if do_plot:
        plt.figure(variable)
        plt.plot(freq, yf.real, label=turbine)
        plt.legend(), plt.grid()
    return amp_fft

def ecart_mediane(v_turbine, turbines, data, variable, H0):
    ## Déclaration des variables ##
    font = {'family' : 'normal', 'weight' : 'normal', 'size' : 12}; matplotlib.rc('font', **font)    
    ## Mise en forme DataFrame ##
    df_variable = data[data['ref_turbine_valorem']==v_turbine[0]][['date_heure', variable]].set_index('date_heure')
    for tur, turbine in enumerate(v_turbine[1:]):
        df_variable = pd.merge(df_variable, data[data['ref_turbine_valorem']==turbine][['date_heure', variable]].set_index('date_heure'),\
             left_index=True, right_index=True, suffixes=(f'_{v_turbine[tur]}',f'_{v_turbine[tur+1]}'))
    df_variable.columns=v_turbine
    ## Calcul du signal médian ##
    mediane_variable = df_variable.median(axis = 1, skipna = True)
    df_indicateur = df_variable.sub(mediane_variable, axis='index').ewm(alpha=0.01).mean()
    df_indicateur = (df_indicateur-df_indicateur[H0[0]: H0[1]].mean()) / df_indicateur[H0[0]: H0[1]].std()
    df_indicateur = df_indicateur[~df_indicateur.index.duplicated(keep='first')]
    ## Affichage ##
    for turbine in turbines:
        plt.plot(df_indicateur[turbine], label=turbine)
    plt.grid(True), plt.title(variable)
    plt.legend(loc = 'lower right')
    return df_indicateur

def couple(v_turbine, data, x, y, z, sample):
    font = {'family' : 'normal', 'weight' : 'normal', 'size' : 10}
    matplotlib.rc('font', **font)
    lines = [[] for x in range(len(v_turbine))]

    _, ax = plt.subplots(num='scatter y(x)')
    plt.subplots_adjust(left=0.15)
    for tur, turbine in enumerate(v_turbine):
        if sample: data_red = data[data['ref_turbine_valorem'] == turbine][[x]+[y]].sample(1000)
        else: data_red = data[data['ref_turbine_valorem'] == turbine][[x]+[y]]
        lines[tur] = ax.scatter(data_red[x], data_red[y], 
            marker='o', alpha=0.5, edgecolors='k',
            label = f'{turbine} R²={round(data_red.corr().loc[x, y], 2)}')#, MI={round(mutual_info_regression(X=data_red[[x, y]].dropna(), y=data_red[y].dropna())[0], 2)}')
    ax.set(xlabel = x, ylabel = y)
    font = {'family' : 'normal', 'weight' : 'normal', 'size' : 15}
    matplotlib.rc('font', **font)
    mplcursors.cursor().connect("add", lambda sel: sel.annotation.set_text(data['date_heure'][sel.index]))
    ax.grid(), ax.legend() #(bbox_to_anchor=(-0.06, 0.5))

    _, axz = plt.subplots(num='scatter y(x) coloriage : z')
    plt.subplots_adjust(left=0.15)
    for tur, turbine in enumerate(v_turbine):
        if sample: data_red = data[[x]+[y]+[z]].sample(1000)
        else: data_red = data[[x]+[y]+[z]]
        lines[tur] = axz.scatter(data_red[x], data_red[y], 
            marker='o', alpha=0.5, edgecolors='k', c=data_red[z], cmap = 'RdYlGn_r')
    font = {'family' : 'normal', 'weight' : 'normal', 'size' : 15}
    matplotlib.rc('font', **font)
    axz.set(xlabel = x, ylabel = y); axz.grid()
    font = {'family' : 'normal', 'weight' : 'normal', 'size' : 8}
    matplotlib.rc('font', **font)

    """fig3D = plt.figure(num='scatter y(x,z)')
    ax3D = fig3D.add_subplot(projection='3d')
    for tur, turbine in enumerate(v_turbine):
        if sample: data_red = data[data['ref_turbine_valorem'] == turbine][[x]+[y]+[z]].sample(1000)
        else: data_red = data[data['ref_turbine_valorem'] == turbine][[x]+[y]+[z]]
        lines[tur] = ax3D.scatter(data_red[x], data_red[y], data_red[z],
            marker='o', alpha=0.5, edgecolors='k',
            label = f'{turbine} R²={round(data_red.corr().loc[x, y], 3)}')
    font = {'family' : 'normal', 'weight' : 'normal', 'size' : 15}
    matplotlib.rc('font', **font)
    font = {'family' : 'normal', 'weight' : 'normal', 'size' : 8}
    matplotlib.rc('font', **font)#"""

    return lines#, fig3D

def coef_rc_model(data, data_models, composant):
    df = pd.DataFrame(index = list(data_models[composant].index), columns=['rc', 'coef'])
    for i, id in enumerate(df.index):
        df.loc[id, 'coef'] = np.abs(data_models[composant].iloc[i, :]).median()
        df.loc[id, 'rc'] = np.abs(round(data.corr().loc[data_models[composant].index[i], data_models[composant].index.name], 3))
    df.plot(kind='bar', secondary_y = 'rc', rot=0), plt.grid()

def matrix(data, classe = ''):

def missing_values(data):
    for turbine in (list(data['ref_turbine_valorem'].unique())):
        # msno matrix #
        fig = plt.figure(turbine)
        ax = fig.add_subplot(1, 1, 1)
        msno.matrix(data[data['ref_turbine_valorem'] == turbine], fontsize=8, color = (random.random(), random.random(), random.random()), labels=True, ax=ax)
        plt.xticks(rotation = 40)
        # msno bar plot #
        fig = plt.figure('missingno bar plot')
        ax = fig.add_subplot(1, 1, 1)
        msno.bar(data, ax=ax, fontsize=8, color = 'k')
        plt.xticks(rotation = 90)

def boxplots_group(data, classe = ''):
    plt.figure(classe)
    data_group = pd.DataFrame()
    for col in data.columns:
        if classe in col and len(data[col].dropna())>0:
            Q1 = (data[col].min() + data[col].median()) / 2
            Q3 = (data[col].max() + data[col].median()) / 2
            ratio_Q1 = (data[data[col] < Q1].count()[col] / len(data[col]))*100
            ratio_Q3 = (data[data[col] > Q3].count()[col] / len(data[col]))*100
            print(col)
            print(f'% données < Q1 = {round(ratio_Q1, 2)}')
            print(f'% données > Q3 = {round(ratio_Q3, 2)} \n')
            data_group = pd.concat([data_group, data[col]], axis = 1)
    print(np.shape(data_group))
    print(data_group)
    if len(data_group)>0: 
        data_group.boxplot()
        plt.title(id)
        plt.xticks(ticks = range(1, len(data_group.columns)+1), labels = data_group.columns, rotation=35, ha='right')
    return data_group.columns

#%% ANALYSE DES INDICATEURS RESIDUELS ##
def indicateur_daily_subplots(TSI, H0, id_nan, v_turbine, v_composant, H1=[]):
    ymin = np.min([TSI[compo][turbine][::144].min() for compo in v_composant for turbine in v_turbine])
    ymax = np.max([TSI[compo][turbine][::144].max() for compo in v_composant for turbine in v_turbine])
    _, ax = plt.subplots(len(v_composant), 1, sharey = True, sharex = True, num='line plot indicateurs résiduels')
    if len(v_composant)>1:
        for c, composant in enumerate(v_composant):
            indicateur_daily(TSI, H0, id_nan, v_turbine, composant, ymin, ymax, ax[c], H1)
    else:
        indicateur_daily(TSI, H0, id_nan, v_turbine, v_composant[0], ymin, ymax, ax, H1)

def indicateur_daily(TSI, H0, id_nan, v_turbine, composant, ymin, ymax, ax, H1=[]):
    font = {'family' : 'DejaVu Sans', 'weight' : 'normal', 'size' : 20}; matplotlib.rc('font', **font)
    ticks_number = 20
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'bisque', 'forestgreen', 'royalblue']
    for tur, turbine in enumerate(v_turbine):

        ## MISE EN FORME DES DONNEES A AFFICHER ##
        if len(H1) > 0:
            tsi = pd.concat([TSI[composant][turbine][H0[0]: H0[1]], TSI[composant][turbine][H1[0]: H1[1]]])
        else:
            tsi = TSI[composant][turbine][H0[0]: H0[1]]
        tsi = tsi.sort_index()
        # distinction des données résiduelles mesurées (m) et imputées (m) #
        tsi_m = pd.Series(data = [np.nan]*len(tsi), index = tsi.index)
        tsi_i = pd.Series(data = [np.nan]*len(tsi), index = tsi.index)
        tsi_m.loc[tsi.index.difference(id_nan[composant][turbine])] = tsi.loc[tsi.index.difference(id_nan[composant][turbine])]
        tsi_i.loc[id_nan[composant][turbine][(id_nan[composant][turbine]>=H0[0]) & (id_nan[composant][turbine]<H0[1])]] = \
            tsi.loc[id_nan[composant][turbine][(id_nan[composant][turbine]>=H0[0]) & (id_nan[composant][turbine]<H0[1])]]
        if len(H1) > 0:
            tsi_i.loc[id_nan[composant][turbine][(id_nan[composant][turbine]>=H1[0]) & (id_nan[composant][turbine]<H1[1])]] = \
            tsi.loc[id_nan[composant][turbine][(id_nan[composant][turbine]>=H1[0]) & (id_nan[composant][turbine]<H1[1])]]
        # échantillonnage #
        h=144
        tsi_m = tsi_m[::h]; tsi_i = tsi_i[::h]; t = tsi.index[::h]
        
        ## AFFICHAGE ##
        ax.plot(tsi_m, '.-', label = 'WT1', color=colors[tur])
        #ax.plot(tsi_i, label = f'{turbine}, imputed TSI', color=colors[tur], linestyle='dashed')        
        # zone H0 #
        ax.add_patch(Rectangle(xy=(mdates.date2num(np.datetime64(H0[0])), ymin), \
            width=mdates.date2num(np.datetime64(H0[1]))-mdates.date2num(np.datetime64(H0[0])), \
            height=np.abs(ymin)+ymax, edgecolor='green', facecolor='palegreen', lw=3, linestyle='dashed'))
        # zone H1 #
        if len(H1) > 0:
            ax.add_patch(Rectangle(xy=(mdates.date2num(np.datetime64(H1[0])), ymin), \
                width=mdates.date2num(np.datetime64(H1[1]))-mdates.date2num(np.datetime64(H1[0])), \
                height=np.abs(ymin)+ymax, edgecolor='red', facecolor='lightcoral', lw=3, linestyle='dashed'))
        # configuration
        ax.set_ylim(ymin, ymax), ax.set_title(f'Health indicator - slow shaft bearing'), ax.set_ylabel('prediction error')
        ax.grid(True), ax.legend(loc = 'lower right'), xaxis_config(pd.Series(data = t), ticks_number)#"""

def KDE_TSI(TSI_H0, v_turbine, s_composant, TSI_H1=[]):
    kde_H1 = [0, 0]
    for composant in list(s_composant.index):
        ## Affichage des TSI (H0) ##
        tsi_H0 = TSI_H0[composant][v_turbine]
        tsi_H0 = pd.DataFrame(data=np.ma.masked_where(tsi_H0 == 1e-5, tsi_H0), index=tsi_H0.index, columns=v_turbine)
        ax = tsi_H0.plot.kde()
        kde_H0 = np.around(tsi_H0.plot.kde().get_lines()[0].get_xydata(), 2)
        ## Affichage des TSI (H1) ##
        if len(TSI_H1) > 0:
            tsi_H1 = TSI_H1[composant][v_turbine]
            tsi_H1 = pd.DataFrame(data=np.ma.masked_where(tsi_H1 == 1e-5, tsi_H1), index=tsi_H1.index, columns=v_turbine)
            tsi_H1.plot.kde(ax = ax, linestyle = 'dashed')
            kde_H1 = np.around(tsi_H1.plot.kde().get_lines()[0].get_xydata(), 2)
        ## Affichage des seuils ##
        if len(TSI_H1) > 0:
            ## Seuil généralisabilité tau_H1 ##
            ax.vlines(tsi_H1.median().values, ymin = 0, ymax = 5, linestyles = 'dotted', colors='k')
            ax.vlines(tsi_H1.mean().values, ymin = 0, ymax = 5, linestyles = 'dashed', colors='k')
        ax.grid(True), ax.set_title(composant)
    return kde_H0, kde_H1

def ROC(TSI, id_nan, H0, H1):
    font = {'family' : 'DejaVu Sans', 'weight' : 'normal', 'size' : 20}
    matplotlib.rc('font', **font)
    df_ROC = pd.DataFrame()
    
    ## CALCUL DES METRIQUES ##
    TSI = TSI.drop(index = id_nan)
    pdf_H0 = func.gaussian_kde_TSI(TSI, H0)
    pdf_H1 = func.gaussian_kde_TSI(TSI, H1)
    for x, tau in enumerate(np.arange(TSI.min(), TSI.max(), 0.1)):
        #s_crossing_time_H1 = func.crossing_time(TSI[H1[0]: H1[1]], tau)
        #if len(s_crossing_time_H1.where(s_crossing_time_H1>0).dropna())>0:
        if tau < TSI.max() or tau > TSI.min():
        #    premiere_intervention = s_crossing_time_H1.where(s_crossing_time_H1 > 0).dropna().index[0]
        #    df_ROC.loc[x, 'detection advance (DA)'] = len(TSI[H1[0]: H1[1]][::144][premiere_intervention: ]) 
            df_ROC.loc[x, 'detection advance (DA)'], df_ROC.loc[x, 'detection date (DD)'] = func.AVD_daily(TSI[H1[0]:H1[1]], id_nan, tau)
        else:
            df_ROC.loc[x, 'detection advance (DA)'], df_ROC.loc[x, 'detection date (DD)'] = 0, H1[1]
        df_ROC.loc[x, 'true positive rate (TPR)'] = 100*(func.AUC(pdf_H1, tau, 'sup')/func.AUC(pdf_H1, pdf_H1.index[-1], 'inf'))
        df_ROC.loc[x, 'false positive rate (FPR)'] = 100*(func.AUC(pdf_H0, tau, 'sup')/func.AUC(pdf_H0, pdf_H0.index[0], 'sup'))
        if (df_ROC.loc[x, 'detection advance (DA)'] >= 7) and (df_ROC.loc[x, 'false positive rate (FPR)'] <=  5):
            df_ROC.loc[x, 'distance'] = np.sqrt((df_ROC.loc[x, 'false positive rate (FPR)']-5)**2 + (df_ROC.loc[x, 'detection advance (DA)']-7)**2)
        df_ROC.loc[x, 'seuil'] = tau
        
    ## AFFICHAGE ##
    color = (random.random(), random.random(), random.random())
    _, ax = plt.subplots(1, 2, num='ROC curves')
    ax[0].plot(df_ROC['false positive rate (FPR)'], df_ROC['true positive rate (TPR)'], ".-", linewidth=2, color=color)
    ax[0].plot(df_ROC.loc[df_ROC['distance'].idxmax(), 'false positive rate (FPR)'], df_ROC.loc[df_ROC['distance'].idxmax(), 'true positive rate (TPR)'], ".-", linewidth=2, markersize=15, label='best TPR, FPR, DA', color=color)
    ax[0].set_xticks(ticks = np.round(np.arange(0, 110, 10), 1), labels = np.round(np.arange(0, 110, 10), 1), rotation = 0, ha = 'right')
    ax[0].set_yticks(ticks = np.round(np.arange(0, 110, 10), 1), labels = np.round(np.arange(0, 110, 10), 1), rotation = 0, ha = 'right')
    ax[0].set_title('TPR(FPR)'), ax[0].set_xlabel('FPR(%)'), ax[0].set_ylabel('TPR(%)'), ax[0].grid(visible=True), ax[0].legend(loc='lower right')
    
    ax[1].plot(df_ROC['false positive rate (FPR)'], df_ROC['detection advance (DA)'], ".-", linewidth=2, color=color)
    ax[1].plot(df_ROC.loc[df_ROC['distance'].idxmax(), 'false positive rate (FPR)'], df_ROC.loc[df_ROC['distance'].idxmax(), 'detection advance (DA)'], ".-", linewidth=2, markersize=15, label='best TPR, FPR, DA', color=color)
    ax[1].set_xticks(ticks = np.round(np.arange(0, 110, 10), 1), labels = np.round(np.arange(0, 110, 10), 1), rotation = 0, ha = 'right')
    ax[1].set_yticks(ticks = np.linspace(0, np.max(df_ROC['detection advance (DA)'])+1, 15, dtype=int), labels = np.linspace(0, np.max(df_ROC['detection advance (DA)'])+1, 15, dtype=int), rotation = 0, ha = 'right')
    ax[1].set_title('DA(FPR)'), ax[1].set_xlabel('FPR(%)'), ax[1].set_ylabel('DA(days)'), ax[1].grid(visible=True), ax[1].legend(loc='lower right')
    ax[1].add_patch(Rectangle(xy=(0, 7), width=5, height=df_ROC['detection advance (DA)'].max()-7, color='g', linewidth = 2, linestyle = 'dashed', fill=True, alpha=0.5))
    return df_ROC
    
#%% HEATMAPS ALARMES/CLASSES ##
def heatmap_config(df, vmin, vmax, cmap, titre):
    ax = sns.heatmap(df, cmap = cmap, cbar=False, vmin = vmin, vmax = vmax, annot = False, fmt=".0f", annot_kws={'size':6})
    for i in range(df.shape[1]+1): ax.axhline(i, color='black', lw=2)
    x = [col[5:10] for col in df.columns]
    plt.title(titre)
    plt.xticks(ticks = range(len(x)), labels =x, rotation = 90, ha = 'right')
    plt.yticks(rotation = 0, ha = 'right'), plt.grid()

def heatmaps_alarmes_flotte(TSI, v_turbine, v_composant, v_parc, data_models, mois, annees, h, do_plot_parc, do_plot_turbine):
    df_classe_flotte = pd.DataFrame(index = v_parc)
    # configuration #
    font = {'family' : 'normal', 'weight' : 'normal', 'size'   : 10}
    matplotlib.rc('font', **font)

    ## CALCUL ##
    for parc in v_parc:
        print(parc)
        # échelle parc #
        df_classe_parc = heatmaps_alarmes_parc(TSI[parc], v_turbine[parc], v_composant[parc], parc, data_models[parc]['seuils'], mois, annees, h, do_plot_parc, do_plot_turbine)
        
        # échelle flotte #
        for d in df_classe_parc.columns:
            df_classe_flotte.loc[parc, d] = df_classe_parc.loc[:, d].max()

    ## AFFICHAGE ##
    plt.figure(f'mois {mois} - flotte')
    heatmap_config(df_classe_flotte, 1, 3, 'YlOrRd', f'mois {mois} - flotte')
    return df_classe_flotte

def heatmaps_alarmes_parc(TSI, v_turbine, v_composant, parc, seuil, mois, annees, h, do_plot_parc, do_plot_turbine): #h = {DAILY,WEEKLY,MONTHLY}
    # configuration #
    font = {'family' : 'normal', 'weight' : 'normal', 'size'   : 10}
    matplotlib.rc('font', **font)
    df_classe_parc = pd.DataFrame(index = v_turbine)
    df_TFA = dict((turbine, pd.DataFrame()) for turbine in v_turbine)
    # sélection de la période paramétrée #
    TSI_c = func.selection_tsi_mois(TSI, v_composant, annees, mois)
    # définition de l'axe des dates #
    dates = list((rrule(h, interval = 1, dtstart = min(TSI_c[v_composant[0]].index), until =  max(TSI_c[v_composant[0]].index))))
    
    for turbine in v_turbine:
        # échelle turbine #
        df_TFA[turbine], df_classe_turbine = heatmap_alarmes_turbine(TSI, turbine, v_composant, seuil, dates)

        ## AFFICHAGE ##
        if do_plot_turbine:
            plt.figure(f'turbine {turbine} - {parc}')
            #plt.subplot(211)
            #heatmap_config(df_TFA[turbine], 0, 1, 'BuPu', f'mois {mois} - parc {parc} - turbine {turbine} - alarmes')
            #plt.subplot(212)
            heatmap_config(df_classe_turbine, 1, 3, 'YlOrRd', f'mois {mois} - parc {parc}  - turbine {turbine} - classes')

        # échelle parc #
        for d in df_classe_turbine.columns:
            df_classe_parc.loc[turbine, d] = df_classe_turbine.loc[:, d].max()

     ## AFFICHAGE : ECHELLE PARC ##
    if do_plot_parc:
        plt.figure(f'mois {mois} - {parc}')
        heatmap_config(df_classe_parc, 1, 3, 'YlOrRd', f'mois {mois} - {parc}')

    return df_classe_parc

def heatmap_alarmes_turbine(TSI, turbine, v_composant, seuil, dates):
    
    df_mu = pd.DataFrame(index = v_composant)
    df_TFA_turbine = pd.DataFrame(index = v_composant)
    df_classe_turbine = pd.DataFrame(index = v_composant)

    for d in range(len(dates)-1):
        
        for composant in v_composant:

            seuil_haut = 2*seuil.loc[composant, '99%']
            data = TSI[composant][turbine][dates[d]:dates[d+1]]
            
            ## calcul du TFA par date ##
            df_TFA_turbine.loc[composant, str(dates[d])[:-9]] = (data.where((data > seuil.loc[composant, '99%'])).count())/data.count()
            df_TFA_turbine = df_TFA_turbine.astype(float)
            
            ## par défaut, classe = NaN ##
            df_classe_turbine.loc[composant, str(dates[d])[:-9]] = np.nan
            
            ## calcul de la moyenne du jour ##
            df_mu.loc[composant, str(dates[d])[:-9]] = data.mean()

            ### Attribution d'une classe si taux d'alarmes élevé ###
            if (df_TFA_turbine.loc[composant, str(dates[d])[:-9]] > 0.5):
                ## d dernières moyennes journalières enregistrées ##
                df_mu_week = df_mu.loc[composant, str(dates[d-min(6, d)]):str(dates[d])]
                ## classification selon df_mu_week ##
                #C3 : début potentiel défaut
                if (len(df_mu_week.where(df_mu_week > seuil_haut).dropna()) > 0) and (len(df_mu_week.where(df_mu_week > seuil_haut).dropna()) <= 2):
                    df_classe_turbine.loc[composant, str(dates[d])[:-9]] = 1 
                #C2 : progression potentiel défaut
                elif (len(df_mu_week.where(df_mu_week > seuil_haut).dropna()) > 2) and (len(df_mu_week.where(df_mu_week > seuil_haut).dropna()) <= 4):     
                    df_classe_turbine.loc[composant, str(dates[d])[:-9]] = 2
                #C1 : défaut confirmé    
                elif len(df_mu_week.where(df_mu_week > seuil_haut).dropna()) > 4:
                    df_classe_turbine.loc[composant, str(dates[d])[:-9]] = 3  
    
    return df_TFA_turbine, df_classe_turbine

def heatmap_multi_résidus(dates, dct_models, YX, YX_learning, data_full, data_learning, H0, H1, ecart_mediane, composant, turbine, df_X):
    v_turbine = list(data_full['brutes']['ref_turbine_valorem'].unique())
    df_valeurs = pd.DataFrame(columns=['X0', 'X1', 'X2', 'X3', 'X4', 'Xr', 'écart médiane', 'y', 'résidu moyen'])
    df_vote = pd.DataFrame(columns=['X0', 'X1', 'X2', 'X3', 'X4', 'Xr', 'écart médiane', 'y', 'vote majoritaire'], index=[str(d)[:-9] for d in dates])
    _, ax = plt.subplots(1, 1, num='comparaison des résidus - line plot')
    for X in df_valeurs.columns:
        ## A) construction des résidus
        # a) écart médian
        if X=='écart médiane':
            df_valeurs[X] = ecart_mediane[turbine]
        # b) y(t)
        elif X=='y':
            data = data_full['standardisées'][data_full['standardisées']['ref_turbine_valorem']==turbine][[dct_models[composant].index.name, 'date_heure']]
            data = data.set_index('date_heure')
            data = data[~data.index.duplicated(keep='first')]
            df_valeurs[X] = data.values
        # c) résidu moyen
        elif X=='résidu moyen':
            df_valeurs['résidu moyen'] = df_valeurs.loc[:, :'écart médiane'].mean(axis=1, numeric_only=True)
        # d) résidus classiques (GFS)
        else:
            dct_models, YX, YX_learning = func.modif_model(dct_models, YX, YX_learning, data_full, data_learning, composant, dct_models[composant].index.name, list(df_X[X]))
            _, TSI, id_nan, _, _, _ = func.TSI_building_and_evaluating(dct_models, v_turbine, [composant], H0, H1, YX, YX_learning)#"""
            df_valeurs[X] = TSI[composant][turbine]
        
        ## B) calcul des décisions de détection & affichage line plot
        # a) écart médian, y(t)
        if (X=='écart médiane') or (X=='y'):
            for d in set(dates):
                if np.abs(df_valeurs[X][d]) > 1:
                    df_vote.loc[str(d)[:-9], X] = 1
                else:
                    df_vote.loc[str(d)[:-9], X] = 0
            df_vote[X] = df_vote[X].replace(np.nan, 0.5)
            if X=='écart médiane':
                ax.plot(df_valeurs[X][::144], linestyle ='dashed', alpha=.6, label=X)
            else:
                plt.figure(dct_models[composant].index.name)
                plt.plot(df_valeurs[X][::144], label=X), plt.legend(), plt.grid()
        # b) résidu moyen
        elif X=='résidu moyen':
            df_vote['vote majoritaire'] = df_vote.loc[:, :'écart médiane'].where(df_vote != 0.5).median(axis=1)
            df_vote['vote majoritaire'] = df_vote['vote majoritaire']
            for d in set(dates):
                if np.abs(df_valeurs[X][d]) > 1:
                    df_vote.loc[str(d)[:-9], X] = 1
                else:
                    df_vote.loc[str(d)[:-9], X] = 0
            ax.plot(df_valeurs['résidu moyen'][::144], 'k', linewidth = 2, label='résidu moyen')
        # c) résidus classiques (GFS)
        else:
            for d in set(dates)-set(list(id_nan[composant][turbine])):
                if np.abs(df_valeurs[X][d]) > 1:
                    df_vote.loc[str(d)[:-9], X] = 1
                else:
                    df_vote.loc[str(d)[:-9], X] = 0
            df_vote[X] = df_vote[X].replace(np.nan, 0.5)
            ax.plot(df_valeurs[X][::144], linestyle ='dashed', alpha=.6, label=X)               
    ax.grid(), ax.legend()
    ax.set_xticks([str(d)[:-9] for d in dates[::14]], [str(d)[:-9] for d in dates[::14]], rotation=90)
    
    ## C) affichage de la heatmap
    plt.figure('comparaison des résidus - heatmap')
    font = {'family' : 'DejaVu Sans', 'weight' : 'normal', 'size' : 10}; matplotlib.rc('font', **font)
    df_1 = df_vote.copy()
    df_1.iloc[:, -2] = float('nan')
    ax = sns.heatmap(df_1.astype(float).T, cmap = 'Greys', cbar=False, annot = False, fmt=".0f", annot_kws={'size':6})
    df_2 = df_vote.copy()
    df_2.iloc[:, :-2] = float('nan')
    sns.heatmap(df_2.astype(float).T, cmap = 'Reds', cbar=False, annot = False, fmt=".0f", annot_kws={'size':6})
    for i in range(df_vote.shape[1]+1): ax.axhline(i, color='black', lw=2)
    
    return df_vote, df_valeurs

#%% LIEN INDICATEURS RESIDUELS / DONNEES 10MIN
def courbe_de_puissance_tsi(data, TSI, v_turbine, debut, fin):
    data_xy = data[['ref_turbine_valorem', 'date_heure', 'vitesse_vent_nacelle', 'puiss_active_produite']]
    data_xy = data_xy[(data_xy['date_heure'] > debut) & (data_xy['date_heure']<=fin)]
    for turbine in v_turbine:
        plt.figure(f'{turbine} {debut}-{fin}')
        xy = data_xy[data_xy['ref_turbine_valorem']==turbine].iloc[:, 2:4]
        z = TSI[turbine][debut: fin]
        sc = plt.scatter(xy.iloc[:min(len(xy), len(z)), 0], xy.iloc[:min(len(xy), len(z)), 1], 
                marker='o', alpha=0.5, edgecolors='k', vmin = -1., vmax = 1., c = z.iloc[:min(len(xy), len(z))], cmap = 'bwr')
        plt.colorbar(sc)
        mplcursors.cursor().connect("add", lambda sel: sel.annotation.set_text(TSI[turbine].index[sel.index]))
        plt.xlabel('vitesse_vent_nacelle'), plt.ylabel('puiss_active_produite')
        plt.xlim(0, 25), plt.ylim(0, 4e6), plt.grid()

def kde_couple_tsi(data, data_models, var, TSI, v_turbine):
    for turbine in v_turbine:
        data_xy = pd.DataFrame({'erreur de prédiction':TSI[turbine].values, var:data[data['ref_turbine_valorem'] == turbine][var].values})
        ## scatter plot pour tous les points, sans KDE ##
        plt.figure(f'{var} {turbine}')
        plt.scatter(data_xy[var], data_xy['erreur de prédiction'], marker='o', alpha=0.5, edgecolors='k')
        plt.hlines(data_models['seuils'].loc['roulement 1 génératrice', '99%'], 
            xmin = min(data[data['ref_turbine_valorem'] == turbine][var]), xmax = max(data[data['ref_turbine_valorem'] == turbine][var]), 
            linestyles = 'dashed', colors='red')
        plt.hlines(data_models['seuils'].loc['roulement 1 génératrice', '1%'], 
            xmin = min(data[data['ref_turbine_valorem'] == turbine][var]), xmax = max(data[data['ref_turbine_valorem'] == turbine][var]), 
            linestyles = 'dashed', colors='black')
        plt.ylim(-1.5, 1.5), plt.grid()
        ## scatter plot pour un échantillon de 10.000 points, avec KDE ##
        g = sns.jointplot(var, "erreur de prédiction", data = data_xy.sample(10000), space=0, color="g", zorder = 0)
        g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)
        plt.ylim(-1.5, 1.5)

#%% EN STAND BY
def jointplot_pm(df_edp, TSI, learning_data, v_turbine, h): #h = {DAILY,WEEKLY,MONTHLY}
    ## PARAMETRISATION ##
    font = {'family' : 'DejaVu Sans', 'weight' : 'normal', 'size'   : 12}; matplotlib.rc('font', **font)
    v_composant = list(learning_data.keys())[:-3]
    data_pm = dict((composant, pd.DataFrame()) for composant in v_composant)

    ## CALCUL HEBDOMADAIRE DES METRIQUES DE PERFORMANCE ##
    for composant in v_composant:
        #tsi = func.selection_tsi_mois(TSI, v_composant, annees, mois)
        dates = list((rrule(h, interval = 1, dtstart = min(TSI[composant].index), until =  max(TSI[composant].index))))

        for tur, turbine in enumerate(v_turbine):
            for d in range(len(dates)-1):
                
                # X : MAME #
                #data_xy.loc[tur*len(dates)+d, 'x'] = data[turbine][dates[d]:dates[d+1]].std()
                if len(df_edp[composant][turbine][dates[d]:dates[d+1]])==0 : #à cause des NaN supprimés
                    data_pm[composant].loc[tur*len(dates)+d, 'MAME'] = np.nan
                else:
                    data_pm[composant].loc[tur*len(dates)+d, 'MAME'] =  mean_absolute_error(df_edp[composant][turbine][dates[d]:dates[d+1]], df_edp[composant]['médiane'][dates[d]:dates[d+1]])
                
                # Y : spécificité #
                #data_xy.loc[tur*len(dates)+d, 'y'] = data[turbine][dates[d]:dates[d+1]].mean()
                data_pm[composant].loc[tur*len(dates)+d, 'spécificité'], _, _ = func.balanced_accuracy(TSI, composant, turbine, learning_data['seuils'], [dates[d], dates[d+1]])
                
                # autres #
                data_pm[composant].loc[tur*len(dates)+d, 'ref_turbine_valorem'] = turbine
                data_pm[composant].loc[tur*len(dates)+d, 'date_heure'] = str(dates[d])[:-9]
    
    ## AFFICHAGE ##
    xmin = np.min([data_pm[compo].loc[:, 'MAME'].min() for compo in v_composant])
    xmax = np.max([data_pm[compo].loc[:, 'MAME'].max() for compo in v_composant])
    ymin = np.min([data_pm[compo].loc[:, 'spécificité'].min() for compo in v_composant])
    ymax = 1.01#np.max([data_pm[compo].loc[:, 'spécificité'].max() for compo in v_composant])
    for composant in v_composant:
        g = sns.jointplot(data = data_pm[composant], x='MAME', y='spécificité', space=0, color="g", zorder = 0, \
            xlim=(xmin, xmax), ylim=(ymin, ymax), hue='ref_turbine_valorem')
        g.plot_marginals(sns.rugplot, color='r', height=.1, clip_on=False)
        g.fig.suptitle(composant)
        cursor = mplcursors.cursor()
        cursor.connect("add", lambda sel: sel.annotation.set_text(data_pm[composant]['date_heure'][sel.index]))
    
    return data_pm

def comparaison_perfs_multimodel(perfs, s_composant, v_turbine, tau_maxH0, tau_minH0, ref, marker, name):
    for composant in list(s_composant.index):
        ## MAME ##
        _, ax = plt.subplots(2, 1, sharex=True, sharey=True, num=composant)
        ax[0].plot(perfs[composant].loc['MAME', :], marker, label = f'modèle {name}')
        ax[0].set_yscale("log"), ax[0].set_ylim(0.9*tau_minH0, 1), 
        ax[0].legend(), ax[0].grid(), ax[0].set_ylabel('MAME (standardisé)')
        if ref:
            # seuils de généralisabilité #
            ax[0].hlines(tau_maxH0, xmin = v_turbine[0], xmax = v_turbine[-1], linestyles = 'dotted', colors='red', label='tau_maxH0')
            ax[0].hlines(tau_minH0, xmin = v_turbine[0], xmax = v_turbine[-1], linestyles = 'dotted', colors='green', label='tau_minH0')
            ax[0].hlines(0.4, xmin = v_turbine[0], xmax = v_turbine[-1], linestyles = 'dashed', colors='red', label='tau_H1')
        
        ## spécificité ##
        ax[1].plot(perfs[composant].loc['faux positifs', :], marker, label = f'modèle {name}')
        if ref:
            ax[1].errorbar(v_turbine, perfs[composant].loc['faux positifs', :], yerr = 144*7, fmt = 'none', elinewidth = 2, capsize = 10, capthick = 2)
        ax[1].yticks(ticks = np.arange(0, 144*30*2, 144*5), labels = np.arange(0, 30*2, 5), ha = 'right')
        ax[1].legend(), ax[1].grid(), ax[1].set_xlabel('turbine'), ax[1].set_ylabel('faux positifs (jours)')

def disponibilité_TSI(YX, parc, v_composant):
    dct_dispo = dict((composant, 0) for composant in v_composant)
    for composant in v_composant:
        dct_dispo[composant] = 100*(round(len(YX[parc][composant].dropna())/len(YX[parc][composant]), 2))
        print(f'disponibilité TSI {composant} parc {parc} : {dct_dispo[composant]}%')
    plt.figure(f'disponibilité TSI parc {parc}')
    plt.bar(range(len(dct_dispo)), list(dct_dispo.values()), align='center')
    plt.ylim(0, 100)
    plt.xticks(range(len(dct_dispo)), list(dct_dispo.keys()))
    return dct_dispo

## performances de détection: 1ère détection à une persistance donnée + persistance sur la période de défaut ##
def performances_detection(TSI, composant, turbine, seuil, nb_jours, debut, fin):
    ## déclaration des variables ##
    data = TSI[composant][turbine][debut:fin]
    duree = 0; indice_detection = 0; premiere_detection = 0
    v_depassement = data[data > seuil[composant][turbine]].index
    ## recherche de la 1ère détection à une persistance de nb_jours ##
    while (duree != f'{nb_jours} days 00:00:00') and (indice_detection + nb_jours*144 <= len(v_depassement)-1):
        duree = str(v_depassement[indice_detection + nb_jours*144]-v_depassement[indice_detection])
        indice_detection += 1
    if len(v_depassement)>0:
        premiere_detection = v_depassement[indice_detection]#"""
    ## calcul de la persistance de l'indicateur sur la période de défaut ##
    persistance = (data.where(data>seuil[composant][turbine]).count())/data.count()
    return persistance, premiere_detection
