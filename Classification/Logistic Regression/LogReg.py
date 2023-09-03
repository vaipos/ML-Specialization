import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import accuracy_score

data = pd.read_csv("sample_data/Social_Network_Ads.csv")
data['Gender'] = data['Gender'].map({'Male' : 0, 'Female' : 1})
drop = ['Purchased']
x_df = data.drop(drop, axis = 1)
x = x_df.values
max = np.max(x, axis=0)
x = x/max
y = data['Purchased']
n = len(x_df.columns)
m = len(y)
w = np.zeros(n)
g = np.zeros(m)
b = 0
z = np.zeros(m)
costadd = 0
loss = 0
epochs = 100
cost = np.zeros(epochs)
alpha = 1.0e-3
epsilon = 1e-15
db = 0
dw = np.zeros(n)
iteration = np.arange(epochs)
purchased_yes = data[data['Purchased'] == 1]
purchased_no = data[data['Purchased'] == 0]


for iter in range(epochs):
  costadd = 0
  db = 0
  dw = np.zeros(n)
  for i in range(m):
    z[i] = np.dot(w,x[i]) + b
    g[i] = 1/(1+(math.e**(-z[i])))
    g[i] = np.clip(g[i], epsilon, 1 - epsilon)
    neg = 1-g[i]
    loss = -((1 - y[i]) * np.log(neg) + y[i] * np.log(g[i]))
    costadd += ((loss)**2)
    db += (loss)/m
    for j in range(n):
      dw[j]+=((loss)*x[i,j])/m
  cost[iter] = costadd/m
  w -= dw * alpha
  b -= db * alpha

print(str(cost[epochs-1]))

plt.scatter(purchased_yes['Age'], purchased_yes['EstimatedSalary'], c='green', label='Purchased: Yes')
plt.scatter(purchased_no['Age'], purchased_no['EstimatedSalary'], c='red', label='Purchased: No')

plt.ylabel("Purchased")

