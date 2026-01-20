import numpy as np
import pandas as pd 
##############################################################################
##This script aim to calculate every indicator useful or choose by the users##
##############################################################################
##############################################################################
#############   FOR EVERY ASSET IN THE PORTFOLIO   ###########################
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
def Return_Portfolio(DataFrame) : 
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

#Relative Strength Index - Portfolio |No Sense|
def RSI(DataFrame): 
    period = 14
    liste_value = np.zeros(period).tolist()
    for i in range(period,len(DataFrame)):
        sum_gain = 0 
        sum_loss = 0
        for date in range(i-period,i+1):
            value = DataFrame[date] - DataFrame[date-1]
            if value > 0 : 
                sum_gain = sum_gain + value
                sum_loss = sum_loss + 0
            elif value <0:
                sum_loss = sum_loss + (-1)*value
                sum_gain = sum_gain + 0
        mean_gain = sum_gain/period
        mean_loss = sum_loss/period
        RS = mean_gain/mean_loss
        RSI = 100 - (100/(1+RS))
        liste_value.append(RSI) 
    DataFrame_RSI = pd.DataFrame({
        "Date" : DataFrame.index,
        "RSI" : liste_value
    })
    return DataFrame_RSI



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


#Relative Strength Index 
def RSI_portfolio(DataFrame): 
    period = 14
    liste_value = np.zeros(period).tolist()
    for i in range(period,len(DataFrame)):
        sum_gain = 0 
        sum_loss = 0
        for date in range(i-period,i+1):
            value = DataFrame["Portfolio"][date] - DataFrame["Portfolio"][date-1]
            if value > 0 : 
                sum_gain = sum_gain + value
                sum_loss = sum_loss + 0
            elif value <0:
                sum_loss = sum_loss + (-1)*value
                sum_gain = sum_gain + 0
        mean_gain = sum_gain/period
        mean_loss = sum_loss/period
        RS = mean_gain/mean_loss
        RSI = 100 - (100/(1+RS))
        liste_value.append(RSI) 
    DataFrame[f"RSI{period}"]=liste_value


