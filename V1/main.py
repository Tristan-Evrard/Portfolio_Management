import pandas as pd 
import config
import Data_Importation as DI
import Data_Vizualisation as DV 
import Distribution_Portfolio as DP



#######################################################################
##This script aim to manage all the program                          ##
#######################################################################

def ask():
    config.path_folder = str(input("What is the path leading to the folder : "))
    config.total_amount = float(input("What is the wage to invest : "))
    config.distribution = str(input("The distribution is uniform ? (y or n) :"))
    if config.distribution == "n" : 
        config.distribution_value = list(input("Quantity of each asset ?"))
    config.confidence_level = float(input("For the VaR, what is your confidence level ?"))

    config.periods_SMA = [
    int(x.strip()) 
    for x in input("How many periods for SMA calculation ? ").split(",")
    if x.strip().isdigit()
    ]

    config.periods_EMA = [
    int(x.strip()) 
    for x in input("How many periods for EMA calculation ? ").split(",")
    if x.strip().isdigit()
    ]
    return ( config.path_folder, config.total_amount, config.confidence_level, config.distribution, config.distribution_value )



######## Run Programm
if __name__ == "__main__": 
    ask() # initialise config.*
    DataFrame_Source = DI.load_data()
    Amount_Each_Value = DI.compute_each_amount(DataFrame_Source)
    DataFrame = DP.Distribution_Portfolio(Amount_Each_Value,DataFrame_Source)
    Confidence_Level = config.confidence_level

    DV.Data_Vizualisation(DataFrame,Confidence_Level,config.periods_SMA,config.periods_EMA)
    