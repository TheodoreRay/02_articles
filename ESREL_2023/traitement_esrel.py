#%% LIBRAIRIES
import pandas as pd
import os
import numpy as np
import math
import openpyxl
import matplotlib
import scipy.stats
import matplotlib.pyplot as plt
import pyarrow as pa
from sklearn.feature_selection import mutual_info_regression
from pyarrow import parquet
from sqlalchemy import create_engine
from tkinter import filedialog, Tk
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, LinearRegression

#%% FONCTIONS
#%% IMPORT ET MISE EN FORME DES DONNEES ##
def sql_request_to_dataframe(v_parc, début, fin):
    engine = create_engine('firebird+fdb://VALEMO:VALEO&M@apps-bdd.valorem.com/S2EV')
    # Définition requête SQL #
    filtre_date = f"AND DATE_HEURE >= '{début}' AND DATE_HEURE < '{fin}'"
    if len(v_parc)>1: 
        SQL_request = f"SELECT * FROM DATA_10MIN WHERE (NOM_PROJET='{v_parc[0]}' OR NOM_PROJET='{v_parc[1]}') {filtre_date} ORDER BY DATE_HEURE ASC"
    else: 
        SQL_request = f"SELECT * FROM DATA_10MIN WHERE NOM_PROJET='{v_parc[0]}' {filtre_date} ORDER BY DATE_HEURE ASC"
    # import données #
    df_data = pd.read_sql_query(SQL_request, engine)
    df_data = df_data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    # pour écriture sous explorateur windows #
    début = début.replace('/', '-')
    fin = fin.replace('/', '-')
    # suppression des caractères spéciaux #
    modele_turbine = ''.join(filter(str.isalnum, df_data['modele_turbine'][0]))[:5]
    if len(v_parc) > 1:
        # unification des références turbine
        L_0 = len(df_data[df_data['nom_projet']==v_parc[0]]['ref_turbine_valorem'].unique())
        L_1 = len(df_data[df_data['nom_projet']==v_parc[1]]['ref_turbine_valorem'].unique())
        for num in range(1, L_1+1):
            df_data.loc[(df_data['ref_turbine_valorem'] == f'T{num}') & (df_data['nom_projet'] == v_parc[1]), 'ref_turbine_valorem'] = f'T{L_0+num}'
        # enregistrement .parquet
        filename = f"{modele_turbine}_{v_parc[0][:5]}_{v_parc[1][:5]}_{début}_{fin}.parquet"
        df_data.to_parquet(filename)
    else: 
        filename = f"{modele_turbine}_{v_parc[0][:5]}_{début}_{fin}.parquet"
        df_data.to_parquet(filename)
    # spécification de la localisation du fichier #
    data_parquet = pa.Table.from_pandas(df_data)
    writer_data_parquet = parquet.ParquetWriter(f'../../../1_data/11_scada_S2EV/parquet/{filename}', schema = data_parquet.schema)
    writer_data_parquet.write_table(data_parquet)
    return df_data, filename

def import_data_auto(format):
    # ouverture de la fenêtre d'intéraction
    root = Tk()
    root.destroy()
    root.mainloop()
    import_file_path = filedialog.askopenfilename()
    # obtention du nom du fichier
    filename = os.path.basename(import_file_path)

    # création du DataFrame
    if format=='parquet':
        df_data = pd.read_parquet(import_file_path)
    if format=='excel':
        df_data = pd.read_excel(import_file_path)    
    #df = SQLContext.read.csv("location", import_file_path)
    return df_data, filename

def import_data(v_parc, v_modele_turbine, v_début, v_fin, is_learning_data):
    # déclaration des variables #
    dct_models = dict((parc, 0) for parc in v_parc)
    dct_data = dict((parc, 0) for parc in v_parc)
    dct_header = dict((parc, 0) for parc in v_parc)
    for x, parc in enumerate(v_parc):
        # import data 10min #
        dct_data[parc] = pd.read_parquet(f'../../../1_data/11_scada_s2ev/parquet/{v_modele_turbine[x]}_{v_parc[x]}_{v_début[x]}_{v_fin[x]}.parquet')
        # import modèles #
        if is_learning_data:
            filepath = f'../../../1_data/12_learning/{v_modele_turbine[x]}_{v_parc[x]}_S2EV.xlsx'
        else:
            filepath = f'../../../1_data/12_learning/TEMPLATE_S2EV.xlsx'
        keys = openpyxl.load_workbook(filepath).sheetnames
        dct_models[parc]= dict((sh, 0) for sh in keys)
        for sh in keys:
            dct_models[parc][sh] = pd.read_excel(filepath, sh)
            dct_models[parc][sh] = dct_models[parc][sh].set_index(dct_models[parc][sh].columns[0], drop=True)
        # création du header #
        dct_header[parc] = {'modele_turbine':v_modele_turbine[x], 'turbines':list(dct_data[parc]['ref_turbine_valorem'].unique())}
    return dct_data, dct_models, dct_header

#%% POST-TRAITEMENT DES DONNEES
def standardisation_data(data_learning, v_data, v_turbine):
    moy = pd.DataFrame(columns = v_turbine)
    std = pd.DataFrame(columns = v_turbine)
    for turbine in v_turbine:
        moy[turbine] = data_learning[data_learning['ref_turbine_valorem']==turbine].mean(numeric_only=True)
        std[turbine] = data_learning[data_learning['ref_turbine_valorem']==turbine].std(numeric_only=True)
    for x in range(len(v_data)):
        data_std = pd.DataFrame()
        for turbine in v_turbine:
            data_turbine = v_data[x][v_data[x]['ref_turbine_valorem']==turbine]
            data_turbine.iloc[:, 2:] = \
                (data_turbine.iloc[:, 2:]-moy[turbine])/std[turbine]
            data_std = pd.concat([data_std, data_turbine])
        v_data[x] = data_std
    return v_data, moy, std

def variable_sorting_lasso(data, v_turbine, y):
    font = {'family' : 'normal', 'weight' : 'normal', 'size' : 10}
    matplotlib.rc('font', **font)
    ## exclusion des variables trop peu disponibles ##
    for var in data.columns[2:]:
        dispo = 100*np.round(data[var].count()/len(data), 2)
        if dispo < 50:
            data = data.drop(columns=var)
    v_coef = pd.DataFrame(columns = data.drop(columns=['ref_turbine_valorem', 'date_heure', y]).columns, index=v_turbine)
    ## calcul des coefficients lasso pour toutes les turbines ##
    for turbine in v_turbine:
        df = data[data['ref_turbine_valorem']==turbine].drop(columns = ['ref_turbine_valorem', 'date_heure']).dropna().copy()
        X_train = df.drop(columns = [y])
        y_train = df[y]
        # calcul des coefs #
        reg_lasso = Lasso(alpha=4e-3, fit_intercept=False).fit(X_train, y_train)
        v_coef.loc[turbine, :] = np.abs(reg_lasso.coef_)
        # mise en forme des coefs #
        v_coef.loc[turbine, :] = pd.Series(data = v_coef.loc[turbine, :], index = X_train.columns)
        v_coef.loc[turbine, :] = v_coef.loc[turbine, :]
    # affichage #
    v_coef = v_coef.median().sort_values()
    #plt.figure('coefficients non nuls par croissant'), v_coef.plot(kind='barh')
    #plt.yticks(ticks = range(len(v_coef.index)), labels = v_coef.index, rotation = 0, ha = 'right'), plt.grid()
    return list(v_coef.index)

def variable_sorting_MI(data, v_turbine, y):
    font = {'family' : 'normal', 'weight' : 'normal', 'size' : 10}
    matplotlib.rc('font', **font)
    v_coef = pd.DataFrame(columns = data.drop(columns=['ref_turbine_valorem', 'date_heure', y]).columns, index=v_turbine)
    for var in data.columns[2:]:
        dispo = 100*np.round(data[var].count()/len(data), 2)
        if dispo < 50:
            data = data.drop(columns=var)
    for turbine in v_turbine:
        df = data[data['ref_turbine_valorem']==turbine].drop(columns = ['ref_turbine_valorem', 'date_heure']).sample(frac=0.1).dropna()
        df_X = df.drop(columns = [y])
        df_y = df[y]
        # calcul des coefs #
        coefs = mutual_info_regression(X=df_X, y=df_y, n_neighbors=3)
        # mise en forme des coefs #
        v_coef.loc[turbine, :] = pd.Series(data = coefs, index = df_X.columns)
    v_coef = v_coef.median().sort_values()
    #plt.figure('coefficients non nuls par ordre décroissant'); v_coef.plot(kind='barh')
    #plt.yticks(ticks = range(len(v_coef.index)), labels = v_coef.index, rotation = 0, ha = 'right'), plt.grid()
    return list(v_coef.index)

#%% MODELES
def gfs_main(data_learning, sortie, v_turbine, L, V):
    resultat = {}
    df_MAE = pd.DataFrame(index = pd.Index(V, name=sortie), columns = [1, 2, 3])
    step = 1
    while step <= L:
        print(f'étape {step}')
        V, resultat, df_MAE = gfs_itération(data_learning, v_turbine, sortie, V, resultat, df_MAE, step)
        step += 1
    return resultat, df_MAE

def gfs_itération(data_learning, v_turbine, sortie, V, resultat, df_MAE, step):
    for v in V:
        # affichage du modèle considéré #
        régresseurs = list(resultat.keys()) + [v]
        print(f'{sortie} = f({régresseurs})')
        # calcul du MAE (médian) #
        v_MAE = [0]*len(v_turbine)
        for x, turbine in enumerate(v_turbine):
            _, v_MAE[x] = model_learning(data=data_learning, turbine=turbine, \
                sortie=sortie, régresseurs=régresseurs, méthode='least squares', test=True)
        df_MAE.loc[v, step] = np.median(v_MAE)
        print(f'MAE (médian) = {df_MAE.loc[v, step]}')
    # sélection de la meilleure variable #
    MAE_opti = df_MAE.loc[:, step].min()
    v_opti = pd.to_numeric(df_MAE.loc[:, step]).idxmin()
    # mise à jour du modèle #
    resultat[v_opti] = MAE_opti
    del V[V.index(v_opti)]
    print(f'dispo = {100*(round(len(data_learning[list(resultat.keys())].dropna())/len(data_learning[list(resultat.keys())]), 2))}%')
    return V, resultat, df_MAE

def model_learning(data, turbine, sortie, régresseurs, méthode, test): # retourne les coefficients de l'équation d'évolution du modèle
    YX = data[['ref_turbine_valorem', 'date_heure'] + régresseurs + [sortie]].dropna()
    YX_train = YX.sample(frac=.8)
    YX_test = YX.loc[list(set(YX.index)-set(YX_train.index))]
    MAE = 1
    if (len(YX_train) > 0) :
        if méthode == 'lasso':
            reg = Lasso(alpha=1e-2).fit(YX_train[YX_train['ref_turbine_valorem'] == turbine][régresseurs],\
                YX_train[YX_train['ref_turbine_valorem'] == turbine][sortie])
            v_coef = reg.coef_
        if méthode == 'least squares':
            reg = LinearRegression().fit(YX_train[YX_train['ref_turbine_valorem'] == turbine][régresseurs],\
                YX_train[YX_train['ref_turbine_valorem'] == turbine][sortie])
            v_coef = reg.coef_
        if test:
            _, y_mes, y_pred, _ = residu_mono(YX_test, turbine, sortie, régresseurs, v_coef)
            MAE = mean_squared_error(y_mes, y_pred)
    else:
        v_coef = [0]*len(régresseurs)
    return v_coef, MAE

def modif_model(dct_models, YX, YX_learning, data_full, data_learning, composant, sortie, variables):
    dct_models[composant] = pd.DataFrame(index=pd.Index(data=variables, name=sortie))
    variables_modele = [dct_models[composant].index.name] + list(dct_models[composant].index.values)
    if all(item in data_learning['standardisées'].columns for item in variables_modele):
        YX[composant] = data_full['standardisées'][['ref_turbine_valorem', 'date_heure'] + list(set(variables_modele))]
        YX_learning[composant] = data_full['standardisées'][['ref_turbine_valorem', 'date_heure'] + list(set(variables_modele))]
    return dct_models, YX, YX_learning

#%% INDICATEURS RESIDUELS : CALCUL
def residu_mono(YX, turbine, sortie, régresseurs, coef):
    # (déjà réalisé en pré-traitement) suppression des dates dupliquées #
    YX = YX[YX['ref_turbine_valorem'] == turbine].set_index('date_heure')
    YX = YX[~YX.index.duplicated(keep='first')].reset_index()
    if len(YX) > 0:
        y_mes = np.array(YX[sortie])
        y_pred = np.array(YX[régresseurs].dot(coef))
        edp_mono = pd.Series(data=y_mes-y_pred, index=YX['date_heure'], name=turbine)
    else:
        y_mes = []
        y_pred = []
        edp_mono = pd.Series(data = [], name = turbine)
    return YX['date_heure'], YX[sortie].values, y_pred, edp_mono

def residu_mono_multi(YX, sortie, régresseurs, v_turbine, df_coef):
    df_edp = pd.DataFrame(columns = v_turbine+['médiane'])
    df_TSI = pd.DataFrame(columns = v_turbine)
    # calcul des résidus mono #
    for turbine in v_turbine:
        _, _, _, df_edp[turbine] = residu_mono(YX, turbine, sortie, régresseurs, df_coef[turbine].values)
    # calcul du résidu médian #
    df_edp['médiane'] = df_edp.iloc[: -1].median(skipna=True, axis=1)
    # calcul des résidus mono-multi #
    for turbine in v_turbine:
        df_TSI[turbine] = df_edp[turbine][df_edp[turbine].index.unique()] - df_edp['médiane']
    return df_edp, df_TSI

#%% INDICATEURS RESIDUELS : EXPLOITATION
def crossing_time(TSI, seuil):
    # déclaration variables #
    s_crossing_time = pd.Series(name='jours de franchissement', dtype='float64')
    tsi = TSI[::144]
    id_crossing = tsi.index[0]
    # détection des périodes de franchissement #
    for x, id in enumerate(tsi.index[1:]): # x : date précédente
        if tsi[id] > seuil:
            if (tsi[tsi.index[x]] <= seuil) or (x == 0): # début de franchissement
                id_crossing = id
                s_crossing_time.loc[id_crossing] = 0
            if tsi[tsi.index[x]] > seuil: # franchissement persistant
                s_crossing_time.loc[id_crossing] += 1
    return s_crossing_time

def AVD_daily(s_vote):
    for v in range(len(s_vote)):
        if s_vote.iloc[v: v+7].sum() < 5:
            d = v
            break
    avd = len(s_vote.iloc[d: ])
    date_de_detection = s_vote.index[d]
    return avd, date_de_detection

def gaussian_kde_TSI(TSI, L):
    x = np.around(np.linspace(min(TSI), max(TSI), 100), 2)
    pdf = pd.Series(index = x, data = scipy.stats.gaussian_kde(TSI[L[0]: L[1]])(x)).sort_index()
    return pdf

def hellinger(TSI, turbine, H0, H1, do_plot):
    # calcul pdfs #
    pdf_H0 = gaussian_kde_TSI(TSI, H0)
    pdf_H1 = gaussian_kde_TSI(TSI, H1)
    # calcul Hellinger #
    sosq = np.sum((np.sqrt(pdf_H0) - np.sqrt(pdf_H1))**2)
    # affichage #
    if do_plot:
        plt.figure(turbine)
        plt.plot(pdf_H0, label='H0')
        plt.plot(pdf_H1, label='H1')
        plt.title(f'H²={round(sosq/math.sqrt(2), 2)}')
        plt.legend(), plt.grid()#"""    
    return pdf_H0, pdf_H1, sosq / math.sqrt(2)

def AUC(pdf, tau, sens):
    AUC = 0
    if sens == 'inf':
        for pdf_i in pdf[: tau]:
            AUC += pdf_i
    if sens == 'sup':
        for pdf_i in pdf[tau: ]:
            AUC += pdf_i
    return AUC
