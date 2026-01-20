import pandas as pd 
import config



def load_data() :
    return pd.read_csv(f"{config.path_folder}", index_col = 0)


def compute_each_amount(DataFrame_Source) :
    #Wage of money to invest
    Total_Amount_To_Invest = config.total_amount

    if config.distribution == "y" : 
        #Total number of Asset
        Total_Asset = len(DataFrame_Source.iloc[0])
        #In the case of a uniformal distribution
        return Total_Amount_To_Invest/Total_Asset
    elif config.distribution=="n": 
        return config.distribution_value
