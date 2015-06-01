import numpy as np
import cPickle as pkl



data_l = []
data_path = "/home/marc/data/"
with open(data_path+'covtype.data') as fp:
    for line in fp:
		tmp_l = [ int(elem) for elem in line.split(',') ]
		data_l.append(tmp_l)


data = np.array(data_l)
np.random.shuffle(data)

quintil = data.shape[0]/5
train_x = data[:quintil*3, :-1]
train_y = (data[:quintil*3,  -1]-1).reshape((-1,1)).astype(int)
valid_x = data[quintil*3:quintil*4, :-1]
valid_y = (data[quintil*3:quintil*4,  -1]-1).reshape((-1,1)).astype(int)
test_x  = data[quintil*4:quintil*5, :-1]
test_y  = (data[quintil*4:quintil*5,  -1]-1).reshape((-1,1)).astype(int)

np.equal(data[:,-1],np.ones(data[:,-1].shape)).sum()
np.equal(data[:,-1],np.ones(data[:,-1].shape)+1).sum()
np.equal(data[:,-1],np.ones(data[:,-1].shape)+2).sum()


dss = [train_x, train_y, valid_x, valid_y, test_x , test_y]
names = ["train_x", "train_y", "valid_x", "valid_y", "test_x", "test_y"]

for ds,name in zip(dss, names):
	f = open(data_path+"COV_"+name+".pkl", "wb")
	pkl.dump(ds,f)