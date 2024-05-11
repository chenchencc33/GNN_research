import os
import time
import Data_Container, Model_Trainer

data_input = Data_Container.DataInput(norm_bool=True, graph_bool=False)
data = data_input.load_data(data_dir='data/JONAS-NYC-16x8-20151024-20160131.npz')
params = dict()
params['C'] = 4
params['H'] = 16
params['W'] = 8
params['batch_size'] = 32
params['device'] = 'cpu'

print(data['mob'].shape) #(4800, 16, 8, 4)
print(data['tcov'].shape) #(4800, 60)
print(data['mask'].shape) #(15, 2)

# get data loader
data_generator = Data_Container.DataGenerator(obs_len=8,
                                                  pred_len=8,
                                                  data_split_ratio=(7,1,2))

data_loader = data_generator.get_data_loader(data=data,
                                            params=params)

#test data
for batch in data_loader['train']:
    x_seq, t_x, t_y, y_seq = batch
    print(x_seq.shape) # [32, 8, 128, 4]
    print(t_x.shape) # [32, 8, 60]
    print(t_y.shape) # [32, 8, 60]
    print(y_seq.shape) # [32, 8, 128, 4]
    break

