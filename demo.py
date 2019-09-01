import dataset
import numpy as np
graph=dataset.Dataset()

batch_size=64
while True:
    mini_batch = graph.sample(batch_size, do_shuffle=False, with_label=False)
    print(np.shape(mini_batch.V1))
    if graph.is_epoch_end:
        break


index=0
idx=graph.generate_samples(do_shuffle=False)

while True:
    if(index>graph.num_nodes):
        break
    if(index+batch_size<graph.num_nodes):
        mini_batch=graph.sample_by_idx(idx[index:index + batch_size])
    else:
         mini_batch=graph.sample_by_idx(idx[index:])
    print(mini_batch)

    index=index+batch_size

    
