### tracker env
import pickle
import numpy as np
import json
    
    
path = '/fsx_ori/yban/dataset/tracker/tapnet/total_data.pickle'

with open(path, 'rb') as f:
    data = pickle.load(f)

trace_matrix = np.zeros((250, 640*2, 360*2))

count = 0
max_seq = 0

for scale, trace, occlusion in data:
    if scale[0][0][0]==640 and scale[0][0][1]==360 and trace.shape[1] == 250:
        for i in range(len(trace)):
            start_x, start_y = None, None
            for j in range(0, trace.shape[1]):
                if (trace[i][j][0]>0 and trace[i][j][1]>0 and trace[i][j][0]<640 and trace[i][j][1]<360) and occlusion[i][j]:
                    if start_x is None or start_y is None :
                        start_x, start_y = trace[i][j]
                        continue
                    x, y = trace[i][j]
                    # if 640+int(x-start_x) >= 1280:
                    #     print('x', x, start_x)
                        # exit()
                    trace_matrix[j, 640+int(x-start_x), 360+int(y-start_y)] += 1
                    count += 1
                    
#                     if trace[i][j][0] > max_seq:
#                         print(trace[i][j][0])
print(count)
# down size the trace matrix by 10,10,10, the new shape is 25, 128, 72, the new element is the sum of the original element
trace_matrix = trace_matrix.reshape(25, 10, 128, 10, 72, 10).sum(axis=(1,3,5))

# convert trace_matrix to list and save it using json
trace_matrix = trace_matrix.tolist()
with open('rope/extract_movement/trace_matrix.json', 'w') as f:
    json.dump(trace_matrix, f)


# plot the heatmap of the number of the points to the distance of the center
# import matplotlib.pyplot as plt

