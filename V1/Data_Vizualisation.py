import matplotlib.pyplot as plt 
import matplotlib.dates as mdates

from Financial_Calculation import * 


def Data_Vizualisation(DataFrame,Confidence_Level,periods_SMA,periods_EMA) :

    DataFrame.index = pd.to_datetime(DataFrame.index, format="%m/%d/%Y")

    #Ploting and Calculation of Value At Risk
    def Portfolio_Value_At_Risk_plot(DataFrame, Confidence_Level,ax) :
        Value_At_Risk = Calc_VaR_Single(DataFrame["Portfolio"], Confidence_Level)
        Distribution = Normalize_Distribution_Single(DataFrame["Portfolio"])

        ax.set_title(f"Value At Risk Representation for Var ={Value_At_Risk}")
        ax.hist(Distribution,bins='auto', alpha=0.7, edgecolor='black')
        ax.axvline(Value_At_Risk, color="red", linewidth = 2, ls = "--")
        ax.legend()




    #Ploting and Calculation of Moving Average
    def Portfolio_SMA_plot(DataFrame, periods,ax) : 

        if isinstance(periods,(int, float)) : 

            DataFrame[f"Portfolio-SMA{periods}"] = SMA(DataFrame["Portfolio"],periods)

            ax.set_title("Simple Moving Average on the portofolio")
            ax.plot(DataFrame.index,DataFrame["Portfolio"], label="Portfolio")
            ax.plot(DataFrame.index,DataFrame[f"Portfolio-SMA{periods}"],label=f"SMA{periods}")
            locator = mdates.AutoDateLocator() 
            formatter = mdates.AutoDateFormatter(locator) 
            ax.xaxis.set_major_locator(locator) 
            ax.xaxis.set_major_formatter(formatter)
        
            ax.legend()

        elif isinstance(periods, (list,tuple)) : 
       
            for i in periods : 
                DataFrame[f"Portfolio-SMA{i}"] = SMA(DataFrame["Portfolio"],i)

        
            ax.set_title("Simple Moving Average on the portofolio")
            ax.plot(DataFrame.index,DataFrame["Portfolio"], label="Portfolio")
            for i in periods :
                ax.plot(DataFrame.index,DataFrame[f"Portfolio-SMA{i}"],label=f"SMA{i}")
            locator = mdates.AutoDateLocator() 
            formatter = mdates.AutoDateFormatter(locator) 
            ax.xaxis.set_major_locator(locator) 
            ax.xaxis.set_major_formatter(formatter)
            ax.legend()

    #Ploting and Calculation of Exponential Moving Average 
    def Portfolio_EMA_plot(DataFrame, periods, ax) : 

        if isinstance(periods,(int, float)) : 

            DataFrame[f"Portfolio-EMA{periods}"] = EMA(DataFrame["Portfolio"],periods)

            ax.set_title("Exponential Moving Average on the portofolio")
            ax.plot(DataFrame.index,DataFrame["Portfolio"], label="Portfolio")
            ax.plot(DataFrame.index,DataFrame[f"Portfolio-EMA{periods}"], label=f"EMA{periods}")
            locator = mdates.AutoDateLocator() 
            formatter = mdates.AutoDateFormatter(locator) 
            ax.xaxis.set_major_locator(locator) 
            ax.xaxis.set_major_formatter(formatter)
            ax.legend()

        elif isinstance(periods, (list,tuple)) : 
       
            for i in periods : 
                DataFrame[f"Portfolio-EMA{i}"] = EMA(DataFrame["Portfolio"],i)

            ax.set_title("Exponential Moving Average on the portofolio")
            ax.plot(DataFrame.index,DataFrame["Portfolio"], label="Portfolio")
            for i in periods :
                ax.plot(DataFrame.index,DataFrame[f"Portfolio-EMA{i}"], label=f"EMA{i}")
            locator = mdates.AutoDateLocator() 
            formatter = mdates.AutoDateFormatter(locator) 
            ax.xaxis.set_major_locator(locator) 
            ax.xaxis.set_major_formatter(formatter)
            ax.legend()


    def plot_graph(periods_SMA,periods_EMA) :
        #Ploting of the graphs 
        #Plotting of every graphs 
        fig ,axes = plt.subplots(1,3, figsize=(8,8))

        Portfolio_Value_At_Risk_plot(DataFrame, Confidence_Level,axes[0])
        Portfolio_SMA_plot(DataFrame, periods_SMA, axes[1])
        Portfolio_EMA_plot(DataFrame, periods_EMA, axes[2])

        fig.autofmt_xdate()

        plt.tight_layout()
        plt.show()
    plot_graph(periods_SMA,periods_EMA)

