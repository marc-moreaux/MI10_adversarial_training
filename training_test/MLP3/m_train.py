import sys
import costAdv
import my_preprocessors as preproc
from pylearn2.config import yaml_parse


def big_print(message):
    print "\n********************************************************"
    print "==> " + message
    print "********************************************************\n"


##########################################################
###  One layer adversarial training
##########################################################
for layer_size in [800, 1000, 1200, 1400, 2000]:
    for learning_eps in [.0, .25]:

        sys.stdout = open("./out_1_"+str(learning_eps)+".txt", "w")

        big_print("train ADVERSARIAL ")
        mlp_yaml = open('mlp1.yaml', 'r').read()
        mlp_hyper_params = {'learning_eps': learning_eps, 'save_path' : '.', 'layer_size': layer_size}
        mlp_yaml = mlp_yaml % mlp_hyper_params
        train_obj = yaml_parse.load(mlp_yaml)
        train_obj.main_loop()


# ##########################################################
# ###  Two layers adversarial training
# ##########################################################
# for learning_eps in [.0, .1, .2, .25,  .3] :

#     sys.stdout = open("./out"+str(learning_eps)+".txt", "a")

#     big_print("train ADVERSARIAL ")
#     mlp_yaml = open('mlp2.yaml', 'r').read()
#     mlp_hyper_params = {'learning_eps': learning_eps, 'save_path' : '.'}
#     mlp_yaml = mlp_yaml % mlp_hyper_params
#     train_obj = yaml_parse.load(mlp_yaml)
#     train_obj.main_loop()



# ##########################################################
# ###  One layer L2 norm training
# ##########################################################
# learning_eps = .0
# sys.stdout = open("./out"+str(learning_eps)+".txt", "a")

# big_print("train ADVERSARIAL ")
# mlp_yaml = open('mlp3.yaml', 'r').read()
# mlp_hyper_params = {'learning_eps': learning_eps, 'save_path' : '.'}
# mlp_yaml = mlp_yaml % mlp_hyper_params
# train_obj = yaml_parse.load(mlp_yaml)
# train_obj.main_loop()



# ##########################################################
# ###   One layer adversarial training on CIFAR10
# ##########################################################
# learning_eps = .0
# sys.stdout = open("./out"+str(learning_eps)+".txt", "a")

# big_print("train ADVERSARIAL ")
# mlp_yaml = open('mlp12.yaml', 'r').read()
# mlp_hyper_params = {'learning_eps': learning_eps, 'save_path' : '.'}
# mlp_yaml = mlp_yaml % mlp_hyper_params
# train_obj = yaml_parse.load(mlp_yaml)
# train_obj.main_loop()

