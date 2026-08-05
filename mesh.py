import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load mesh nodes (grid points)
df = pd.read_csv("mesh_data.csv", skiprows=1, header=None)
x = df.iloc[:, 0].values
y = df.iloc[:, 1].values

# Load cell centers
df2 = pd.read_csv("cell_data.csv", skiprows=1, header=None)
x2 = df2.iloc[:, 0].tolist()
y2 = df2.iloc[:, 1].tolist()

#Load inlet/outlet
df3 = pd.read_csv("inlet_data.csv", skiprows=1, header=None)
x3 = df3.iloc[:, 0].tolist()
y3 = df3.iloc[:, 1].tolist()

df4 = pd.read_csv("outlet_data.csv", skiprows=1, header=None)
x4 = df4.iloc[:, 0].tolist()
y4 = df4.iloc[:, 1].tolist()

#Load walls
df5 = pd.read_csv("top_wall.csv", skiprows=1, header=None)
x5 = df5.iloc[:, 0].tolist()
y5 = df5.iloc[:, 1].tolist()

df6 = pd.read_csv("bottom_wall.csv", skiprows=1, header=None)
x6 = df6.iloc[:, 0].tolist()
y6 = df6.iloc[:, 1].tolist()

num_cols = 100
num_rows = 100


X = x.reshape(num_rows, num_cols)
Y = y.reshape(num_rows, num_cols)


plt.plot(X, Y, color='black', linewidth=1)
plt.plot(X.T, Y.T, color='black', linewidth=1)

plt.scatter(x3, y3, color='red', zorder=5, label='inlet')
plt.scatter(x4, y4, color='blue', zorder=5, label='outlet')
plt.scatter(x5, y5, color='orange', zorder=5, label='top_wall')
plt.scatter(x6, y6, color='purple', zorder=5, label='bottom_wall')

plt.legend()


plt.show()