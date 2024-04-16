import pandas as pd
import os
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