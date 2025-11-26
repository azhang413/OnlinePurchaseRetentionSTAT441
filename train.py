from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import pandas as pd 

# fetch dataset 
online_shoppers_purchasing_intention_dataset = fetch_ucirepo(id=468) 
  
# data (as pandas dataframes) 
X = online_shoppers_purchasing_intention_dataset.data.features 
y = online_shoppers_purchasing_intention_dataset.data.targets 
  
# variable information 
print(online_shoppers_purchasing_intention_dataset.variables)

print(X.head())

for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        plt.figure()
        X[col].plot(kind='hist', title=col)
        plt.xlabel(col)
        plt.show()
