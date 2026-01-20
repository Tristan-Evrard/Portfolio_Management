import numpy as np
import pandas as pd 
##############################################################################
##This script aim to calculate every indicator useful or choose by the users##
##############################################################################


##############################################################################
##              FOR EVERY ASSET IN THE PORTFOLIO                            ##
##############################################################################
#Correlation Calculation 
def Correlation(DataFrame) : 
    return DataFrame.corr(method="pearson")

#Variance Calculation 
def Variance(DataFrame):
    return DataFrame.var(ddof=0)

#Standard Deviation Calculation
def Standard_Deviation(DataFrame):
    return DataFrame.std(ddof=0)

#Value At Risk Calculation
def Calc_VaR(DataFrame, confidence_level) : 
    Noms = DataFrame.columns
    value_liste = []
    for i in Noms : 
        returns = DataFrame[i].pct_change().dropna()
        #Calcul de la VaR
        VaR_historical = np.percentile(returns, (1 - confidence_level) * 100)
        value_liste.append(VaR_historical)
    
    Tableau = pd.DataFrame({'Name' : Noms,
                            'Value - VaR' : value_liste
    })
    Tableau = Tableau.set_index('Name')
    return Tableau 

#Return Calculation 
def Return(DataFrame) : 
    names = DataFrame.columns
    value_list = []
    for i in names : 
        Value = float((DataFrame[i][-1]-DataFrame[i][0])/DataFrame[i][-1])
        Value  = round(Value,4)
        value_list.append(Value)

    Tableau = pd.DataFrame({
        'Name' : names,
        'Return' : value_list
    })
    Tableau = Tableau.set_index('Name')
    return Tableau

##############################################################################
##              FOR ONE ASSET IN THE PORTFOLIO                              ##
##############################################################################

#Variance Calculation 
def Variance_Single(DataFrame):
    return DataFrame.var(ddof=0)

#Standard Deviation Calculation
def Standard_Deviation_Single(DataFrame):
    return DataFrame.std(ddof=0)

#Value At Risk Calculation
def Calc_VaR_Single(DataFrame, confidence_level) : 
    returns = DataFrame.pct_change().dropna()
    #Calcul de la VaR
    VaR_historical = np.percentile(returns, (1 - confidence_level) * 100)

    return VaR_historical

#Normalization of Data
def Normalize_Distribution_Single(DataFrame) :
    return DataFrame.pct_change().dropna()

#Return Calculation
def Return_Single(DataFrame) : 
    Value = float((DataFrame.iloc[-1]-DataFrame.iloc[0])/DataFrame.iloc[-1])
    Value  = round(Value,4)
    return Value    

#Simple Moving Average Calculation 
def SMA(DataFrame, periods):
    periods = int(periods) 
    if periods <= 0: 
        raise ValueError(f"SMA period must be > 0, got {periods}")
    return DataFrame.rolling(window=periods, min_periods=1).mean()

#Exponential Moving Average Calculation 
def EMA(DataFrame,periods) : 
    return DataFrame.ewm(span =int(periods), adjust = False).mean()


