import Data_Importation as DI 
import Financial_Calculation as DF
import matplotlib.pyplot as plt 
list_var = DI.load_data()[3].pct_change().dropna().std()
list_return = []

df = DI.load_data()[3]


for i in range(0,len(df.iloc[0])):
    val_return = DF.Return_Single(df.iloc[:,i])
    list_return.append(val_return)

print(list_return)
print(list_var)

plt.figure()
plt.xlim((0,0.75))
plt.ylim((0,0.5))

plt.scatter(list_return,list_var)

plt.xlabel("Std")
plt.ylabel("Return")


plt.show()