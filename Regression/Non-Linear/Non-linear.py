# All necessary imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score # Algorithm for accuracy calculation

# Alright so...
# Start off with the basic local variable setup
data = pd.read_csv("sample_data/Fish.csv") # we got the data file
xdrop = ["Weight","Species"] # we got the columns we don't want in an array from feature engineering
x_train = data.drop(xdrop, axis = 1) # Now we have the final product
y_train = data["Weight"] # This is the y or the answers
x = x_train.values # Array format instead of a data frame
y = y_train.values/y_train.max() # We scale to make everything 0 to 1
m = len(y) # How many rows
n = len(x_train.columns) # How many columns
epochs = 900 # How many iterations for Gradient Descent
iter = np.arange(epochs) # X - axis for the Cost v.s. Iteration graph
w = np.zeros(n) # weight, the slope, the annoying one
b = 10 # The random starter bias or y-intercept
xmax = np.max(x, axis=0) # The max value for each column 
x /= xmax # we divide the max value of each column to each column, to scale to 0 - 1
f_wb = np.zeros(m) # Our beautiful predicted model
totCost = 0 # Cost adder ... the variable is not intuitive ... my bad
cost = np.zeros(epochs) # Y - axis for the cost v.s. Iteration graph
dw = np.zeros(n) # derivative of w for the gradient descent
db = 0 # derivative for b again for the gradient descent
alpha = 1.0e-1 # learning rate.... the size of the step you take during gradient descent

# Now the actual stuff: 
# Gradient Descent, Loss, Cost, and Regression 
for i in range(epochs): 
  # we reset for every new iteration
  dw = np.zeros(n) 
  db = 0 
  totCost = 0
  # We go through each row and find loss/cost for each of the predicted compared to true values
  for row in range(m):
    f_wb[row] = (np.dot(x[row]**2,w)+b) # Quadratic Regression model 
    totCost += ((f_wb[row] - y[row])**2)/(2*m) # Cost for that predicted model
    db += (f_wb[row] - y[row])/m # Gradient descent derivative formula for bias
    for col in range(n):
      dw[col] += ((f_wb[row] - y[row]) * x[row,col])/m # Each weight in each row vector and calculate derivative
  cost[i] = totCost # Total cost for that iteration
  w -= alpha * dw # new weight and get it better/less cost with each iteration 
  b -= alpha * db # same thing 

# we do be displaying and graphing here:
print("cost in depth = " + str(cost[epochs-1])) # final cost of last w and b 

# Make a graph for each feature and see how the model got it
for i in range(n):
  plt.figure(i) # putting on the same graph
  plt.scatter(x[:,i],y, label ='true') # x is all the column values in that feature
  plt.scatter(x[:,i],f_wb, label = "predicted") # predicted model 
  plt.title(str(x_train.columns[i])) # naming it the feature's name
  plt.legend()
plt.figure(n+1) 
plt.plot(iter,cost) # To see how many iterations we need. When the graph is flat that's how many iterations
plt.title("cost v iteration")
print("The accuracy of the model: " + str(r2_score(y, f_wb))) # Accuracy of the model
