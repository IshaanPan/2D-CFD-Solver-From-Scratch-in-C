import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri 

df=pd.read_csv("field-446.csv", skiprows=1)
x=df.iloc[:,0]
y=df.iloc[:,1]
vel_u=df.iloc[:,4]
vel_v=df.iloc[:,5]


velocity_magnitude = np.sqrt(vel_u**2 + vel_v**2)

plt.figure(figsize=(12, 6))

# 1. Create the standard triangulation
triang = tri.Triangulation(x, y)

# 2. Get the coordinates of the triangles
x_tris = triang.x[triang.triangles]
y_tris = triang.y[triang.triangles]

# 3. Calculate the length of all 3 edges for every triangle
l1 = np.hypot(x_tris[:,0] - x_tris[:,1], y_tris[:,0] - y_tris[:,1])
l2 = np.hypot(x_tris[:,1] - x_tris[:,2], y_tris[:,1] - y_tris[:,2])
l3 = np.hypot(x_tris[:,2] - x_tris[:,0], y_tris[:,2] - y_tris[:,0])

# 4. Find the maximum edge length per triangle
max_edge = np.max(np.column_stack((l1, l2, l3)), axis=1)

# 5. Mask out triangles that are too big (crossing the empty space)
# TWEAK THIS NUMBER: If it still draws a line, make it smaller. If it cuts out real data, make it bigger.
threshold = 0.03
triang.set_mask(max_edge > threshold)

# 6. Plot the smooth masked contour
heatmap = plt.tricontourf(triang, velocity_magnitude, levels=50, cmap='viridis')

plt.colorbar(heatmap, label='Velocity Magnitude')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Velocity Magnitude Heatmap')
plt.axis('equal')
plt.tight_layout()
plt.show()