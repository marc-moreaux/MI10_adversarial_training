import sys
import costAdv
from pylearn2.config import yaml_parse
import os.path



# preptrain layer 1
bin_train = [2,5,25,100,200,800,1200,1400,1800,2000]

for nhid in [400]:

    if not os.path.isfile('./mem/dae_l1_bin_'+str(nhid)+'.pkl'):
        layer1_yaml = open('dae1_bin.yaml', 'r').read()
        hyper_params_l1 = {'train_stop' : 50000,
                           'batch_size' : 100,
                           'monitoring_batches' : 5,
                           'nhid' : nhid,
                           'max_epochs' : 100,
                           'save_path' : '.'}
        layer1_yaml = layer1_yaml % (hyper_params_l1)
        train = yaml_parse.load(layer1_yaml)
        train.main_loop()


    for eps_adv in [.3]:
        # mlp layer 2
        mlp_yaml = open('mlp1_bin.yaml', 'r').read()
        hyper_params_mlp = {'train_stop' : 50000,
                            'valid_stop' : 60000,
                            'batch_size' : 100,
                            'nhid' : nhid,
                            'eps_adv' : eps_adv,
                            'max_epochs' : 1500,
                            'save_path' : '.'}
        mlp_yaml = mlp_yaml % (hyper_params_mlp)
        train = yaml_parse.load(mlp_yaml)
        train.main_loop()

