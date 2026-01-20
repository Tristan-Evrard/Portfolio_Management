import config 
import cvxpy as cp
import numpy as np 
import Data_Importation as DI 
def Optimization(DataFrame): 
    Total_Amount_To_Invest = config.total_amount


    nb_asset = DataFrame.shape[1]
    
    DataFrame_bis = DataFrame.pct_change().dropna()

    cov_dataframe = DataFrame_bis.cov()
    

####This part has been wrote by AI###################################
    weight_asset = cp.Variable(nb_asset)

    objective = cp.Minimize(cp.quad_form(weight_asset,cov_dataframe))

    constraints = [
        cp.sum(weight_asset)==1,  #Sum of wheight must be one
        weight_asset >= 0.01,                 #No short selling
        weight_asset <= 0.10               #Max lenght: 5%
    ]   

    #Opimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve()
#####################################################################
    return (np.round(weight_asset.value,4)*Total_Amount_To_Invest).astype(float).tolist()
        



def compute_each_amount(DataFrame_Source) :
    #Wage of money to invest
    Total_Amount_To_Invest = config.total_amount

    if config.distribution == "n" : 
        
        if config.optimization_value == "y":
            return Optimization(DataFrame_Source)
        if config.optimization_value =="n":
            return config.distribution_value
    
    elif config.distribution=="y": 
        #Total number of Asset
        Total_Asset = len(DataFrame_Source.iloc[0])
        #In the case of a uniformal distribution
        return Total_Amount_To_Invest/Total_Asset


        