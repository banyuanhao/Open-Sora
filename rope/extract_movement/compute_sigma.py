import json
import numpy as np
path = 'rope/extract_movement/trace_matrix.json'
matrix = json.load(open(path))
matrix = np.array(matrix)
print(matrix.shape)
frame, height, width = matrix.shape
x_ave = np.mean(matrix, axis=1)
y_ave = np.mean(matrix, axis=2)

x_dev_lst = np.zeros(frame)
y_dev_lst = np.zeros(frame)
# 128, 72
# 64 64 
# the x_ave records the count of the points in the x direction in each frame, please compute the deviation for each frame
for i in range(frame):
    mean_x = np.sum(x_ave[i]*np.arange(width))/np.sum(x_ave[i])
    mean_y = np.sum(y_ave[i]*np.arange(height))/np.sum(y_ave[i])
    # print(mean_x, mean_y)
    x_dev = np.sum((x_ave[i]*(np.arange(width)-mean_x)**2))/np.sum(x_ave[i])
    y_dev = np.sum((y_ave[i]*(np.arange(height)-mean_y)**2))/np.sum(y_ave[i])
    print(x_dev, y_dev)
    x_dev_lst[i] = x_dev
    y_dev_lst[i] = y_dev

import matplotlib.pyplot as plt
# least square fitting for the first 10 elements of x_dev_lst and y_dev_lst and plot the result
x = np.arange(10)
A = np.vstack([x, np.ones(10)]).T
m, c = np.linalg.lstsq(A, x_dev_lst[:10], rcond=None)[0]
plt.plot(x, m*x + c, 'r')
m, c = np.linalg.lstsq(A, y_dev_lst[:10], rcond=None)[0]
plt.plot(x, m*x + c, 'b')




# x = np.arange(frame)
# A = np.vstack([x, np.ones(frame)]).T
# m, c = np.linalg.lstsq(A, x_dev_lst, rcond=None)[0]
# plt.plot(x, m*x + c, 'r')
# m, c = np.linalg.lstsq(A, y_dev_lst, rcond=None)[0]
# plt.plot(x, m*x + c, 'b')

plt.plot(x_dev_lst)
plt.plot(y_dev_lst)
plt.savefig('dev.png')

# the y_ave records the count of the points in the y direction in each frame, please compute the deviation for each frame
