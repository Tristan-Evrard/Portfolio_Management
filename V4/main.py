import pandas as pd 
import numpy as np
import config
import Data_Importation as DI
import Data_Visualization as DV 
import Distribution_Portfolio as DP
import Portfolio_Optimization as PO
from Efficient_Fronter import Efficient_Frontier


#######################################################################
##This script aim to manage all the program                          ##
#######################################################################

def ask():

    config.path_folder = str(input("What is the path leading to the folder : ")) #What folder is use
    config.total_amount = float(input("What is the wage to invest : "))#What wage of money will be use
    config.distribution = str(input("The distribution is uniform ? (y or n) :"))#Use a uniform distribution 

    if config.distribution == "n" : #No uniform distribution
        config.optimization_value = str(input("Do you want to have a specific optimization ?"))#Want a Portofolio Optimization
        if config.optimization_value =="y": #Yes
             pass
        elif config.optimization_value =="n": #No 
            config.distribution_value = list(input("Quantity of each asset ?"))#What is the distribution

    config.confidence_level = float(input("For the VaR, what is your confidence level ?"))#What is the confidence level

    #Configure the SMA calculation
    config.periods_SMA = [
    int(x.strip()) 
    for x in input("How many periods for SMA calculation ? ").split(",")
    if x.strip().isdigit()
    ]

    #Configure the EMA calculation
    config.periods_EMA = [
    int(x.strip()) 
    for x in input("How many periods for EMA calculation ? ").split(",")
    if x.strip().isdigit()
    ]
    
    #Configure the Risk Free Rate
    config.Risk_Free = str(input("Is there a Risk Free asset in your assets ? (y/n)"))
    if config.Risk_Free == "n":
        config.Risk_Free = 0.0
    elsif congif.Risk_Free =="y": 
        config.Risk_Free = float(input("What is it rate ?"))

    return ( config.path_folder, config.total_amount, config.confidence_level, config.distribution, config.distribution_value )



######## Run Programm
if __name__ == "__main__": 
    
    #ask() # initialise config.*

    #If there is O/H/L/C/V
    if len(DI.load_data()) == 5 :
        df_open = DI.load_data()[0]
        df_high = DI.load_data()[1]
        df_low = DI.load_data()[2]
        df_close = DI.load_data()[3]
        df_volume = DI.load_data()[4]

    #If there is only the Close value
    if len(DI.load_data()) != 5   : 
        df_open = None
        df_high = None
        df_low = None
        df_close = DI.load_data()
        df_volume = None

    #Calcul the distribution of the portfolio according to the previous answer
    Amount_Each_Value = PO.compute_each_amount(df_close)

    #Caclul the distribution of the portfolio for each time value
    DataFrame = DP.Distribution_Portfolio(Amount_Each_Value,df_open,df_high,df_low,df_close)

    #If there is O/H/L/C in the dataframe
    if len(DI.load_data()) == 5 :
            df_portfolio_open = DataFrame[0]
            df_portfolio_high = DataFrame[1]
            df_portfolio_low = DataFrame[2]
            df_portfolio_close = DataFrame[3]
    #Else there is the Close value of the portfolio
    else : 
         df_portfolio_close = DataFrame
    
    Names = None
    word_delete = {"_Close"}
    if len(DI.load_data()) == 5 :
        Names = [name.replace("_Close","") for name in df_close.columns]
    else : 
        Names = df_close.columns

    DataFrame_Amount_Each_Value = pd.DataFrame({
         "Names" : Names,
         "Quantity per asset" : DataFrame[4]
    })
    
    #Set the ploting
    DV.Data_Visualization(df_portfolio_close,DataFrame_Amount_Each_Value,
                          Confidence_level =config.confidence_level,periods_SMA =config.periods_SMA,
                          periods_EMA = config.periods_EMA,
                          RSI = config.RSI).plot_graph()
    
    
    # ── Frontière efficiente de Markowitz ─────────────────────────────────────
    # Calcul des poids à partir des montants alloués
    amounts = np.array(Amount_Each_Value, dtype=float)
    weights = amounts / amounts.sum()

    Efficient_Frontier(
        df_close           = df_close,
        portfolio_weights  = weights,
        n_simulations      = 4000,
        trading_days       = 252,
        rf                 = config.Risk_Free,
    ).plot()
