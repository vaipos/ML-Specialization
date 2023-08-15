# Linear Regression with Gradient Descent & Cost
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Kaggle Simple Dataset for Linear Regression
data =  pd.read_csv('sample_data/Salary_dataset.csv')
x_train = data.iloc[:,1:2].values
y_train = data.iloc[:,2:].values
# Scaling to make Gradient Descent work proportionally
# x -> 0 - 10.6 -> 0 - 1
# y -> 0 - 121873 -> 0 - 1
x = x_train/10.6
y = y_train/121873

# Local Variables
m = len(x)
scale = 121873
epochs = 500 # Number of iterations... this is so much better than some lady's attempt at this dataset
iter = np.arange(epochs)
ans = np.zeros(epochs)
f_wb = np.zeros(m)
w = 1000/scale
b = 1000/scale
costW = 0
costB = 0
alpha = 1.0e-1 # Learning Rate ... step size for Gradient Descent on the cost v W graph


# Linear Regression Model (y = mx + b)
# New Terminology:
# slope = w -> weight
# y-intercept = b -> bias
def lin(x, w, b):
  return (w*x) + b

# Gradient Descent Algorithm
for i in range(epochs):
  cost = 0
  costW = 0
  costB = 0
  # Cost + Gradient Descent
  for j in range(m):
    f_wb[j] = lin(x[j],w,b)  # Linear Reg Base Formula
    cost += ((f_wb[j]-y[j])**2)/(2*m) # Cost function for the Cost v Iterations Graph
    costW += ((f_wb[j] - y[j])*x[j])/m  # Derivative or slope check for w from the cost v w graph
    costB += (f_wb[j] - y[j])/m # Derivative for B formula
    # Summation for all values of x

  # Previous weight to find a better weight with smaller derivative
  w = w - (costW*alpha)
  b = b - (costB*alpha)
  ans[i] = cost # hold cost of that w's iteration
print("w = " + str(w*scale)) # Scaled back to accurate weight for dataset
print("b = " + str(b*scale)) # sametinggg
print("Cost in depth: " + str(ans[epochs-1]))


# It do be graphing time besties
# The Regression Plot:
# See how our predict Weight and Bias look compared to the data points
# Made it organized and learnt a couple of the library function to do so
plt.figure(1)
plt.scatter(x_train, y_train, c='r',label='Data points')
plt.plot(x_train, f_wb*scale, c='b',label='Model')
plt.xlabel('Years of Experience')
plt.ylabel('Salary $')
plt.title('Linear Regression Plot')
plt.legend()

# The Cost V Iteration Plot
# This plot helps see how many epochs is necessary to get a very low cost
# When the graph shows a slope of almost 0 thats the best count of epochs
plt.figure(2)
plt.plot(iter, ans, c='b')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost V Iterations')


