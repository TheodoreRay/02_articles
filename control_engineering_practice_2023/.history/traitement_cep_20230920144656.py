import pandas as pd
import os
import numpy as np
from tkinter import filedialog, Tk

def import_data_auto(format, sheet_names=[]):
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
        if len(sheet_names)>0:
            df_data = dict((key, 0) for key in sheet_names)
            for sheet in sheet_names:
                df_data[sheet] = pd.read_excel(import_file_path, sheet)    
        else:
            df_data = pd.read_excel(import_file_path)    
    #df = SQLContext.read.csv("location", import_file_path)
    return df_data, filename

def alert(df_epsi, tau):
    alertCount = 0
    dates = pd.DataFrame(columns=['début', 'fin', 'durée'])
    durée = 0
    df_epsi = df_epsi.replace(np.nan, 0.5)

    ## comptage du alertCount et stockage des dates de début et fin ##
    for x, date in enumerate(df_epsi.index):
        ## indicateur supérieur au seuil ##
        if (df_epsi[x] > tau):
            if durée == 0:
                dates = pd.concat([dates, pd.DataFrame({'début':date, 'fin':date, 'durée':durée},index=[len(dates)])])
            durée += 1
            
        ## indicateur nul ou indisponible ##
        if (df_epsi[x] <= tau) or (x==len(df_epsi)-1):
            if durée > 0:
                if durée >= 144:
                    alertCount += 1
                    dates.loc[len(dates)-1, ['fin','durée']] = [date, durée]
                else:
                    dates = dates.drop([len(dates)-1])
            durée = 0
    
    return dates, alertCount