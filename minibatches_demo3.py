import numpy as np
from dataset import minibatches
inputs=np.array([1, 2, 3, 4])
targets=np.array([1, 2, 3, 4])
for c in range(5):
    print(str(c)+":c")
    creatorGenerate=minibatches(inputs,targets,batch_size=1)
    for i,j in creatorGenerate:
        print(i,j)



