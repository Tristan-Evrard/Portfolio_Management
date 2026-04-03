import pandas as pd 
import config 



def load_data() :
    df = pd.read_csv(f"{config.path_folder}", index_col = 0)
    
    Open = []
    High = []
    Low = []
    Close = []
    Volume = []

    df_col = df.columns
    for df_col_value in df_col :  
        if "Open" in df_col_value : 
            Open.append(df_col_value)
        if "High" in df_col_value : 
            High.append(df_col_value)
        if "Low" in df_col_value :
            Low.append(df_col_value) 
        if "Close" in df_col_value : 
            Close.append(df_col_value)
        if "Volume" in df_col_value : 
            Volume.append(df_col_value)
        
    df_open = df[Open]
    df_high = df[High]
    df_low = df[Low]
    df_close = df[Close]
    df_volume = df[Volume]

    if Close == [] : 
        return df
    else : 
        return df_open,df_high,df_low,df_close,df_volume
