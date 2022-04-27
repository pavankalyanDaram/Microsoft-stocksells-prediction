#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.style.use('fivethirtyeight')


# In[2]:


data = pd.read_csv("C:\\Users\\Pavan Kalyan\\MSFT.csv")
print(data.head(10))


# In[3]:


data.describe()


# In[4]:


plt.figure(figsize=(10, 4))
plt.title("Microsoft Stock Prices")
plt.xlabel("Date")
plt.ylabel("Close")
plt.plot(data["Close"])
plt.show()


# In[5]:


data ['Adj Close'].plot()


# In[6]:



X=data.drop(['Adj Close'],axis=1)
X=X.drop(['Close'],axis=1)
X.corrwith(data['Adj Close']).plot.bar(
        figsize = (20, 10), title = "Correlation with Adj Close", fontsize = 20,
        rot = 90, grid = True)


# In[7]:


print(data.corr())
sns.heatmap(data.corr())
plt.show()


# In[8]:


x = data[["Open", "High", "Low"]]
y = data["Close"]
x = x.to_numpy()
y = y.to_numpy()
y = y.reshape(-1, 1)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)


# In[9]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
data = pd.DataFrame(data={"Predicted_Rate": ypred})
print(data.head(10))


# In[10]:


plt.figure(figsize=(10, 4))
plt.title("Microsoft Stock Prices Prediction over time")
plt.xlabel("Date")
plt.ylabel("Predicted rate")
plt.plot(data)
plt.show()


# In[13]:


dataset_test = pd.read_csv('C:\\Users\\Pavan Kalyan\\MSFT.csv')
real_Sales = dataset_test.iloc[:, 1:2].values


# In[14]:


plt.figure(figsize=(12,6))
plt.plot(real_Sales, label = 'Real Price')
plt.plot(data, label = 'Pred Price')
plt.xlabel('Date', size=12)
plt.ylabel('Microsoft stock prices', size=12)
plt.title('Microsoft stock price prediction',size=15)
plt.legend()

