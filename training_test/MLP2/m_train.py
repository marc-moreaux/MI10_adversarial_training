import sys
import costAdv
import draw_images
import my_preprocessors as preproc
from pylearn2.config import yaml_parse


def big_print(message):
    print "\n********************************************************"
    print "==> " + message
    print "********************************************************\n"


#
# Kind of main thingy
#
for learning_eps in [.0, .1, .2, .25,  .3] :

    sys.stdout = open("./out"+str(learning_eps)+".txt", "a")

    big_print("train ADVERSARIAL ")
    mlp_yaml = open('mlp1.yaml', 'r').read()
    mlp_hyper_params = {'learning_eps': learning_eps, 'save_path' : '.'}
    mlp_yaml = mlp_yaml % mlp_hyper_params
    train_obj = yaml_parse.load(mlp_yaml)
    train_obj.main_loop()


for learning_eps in [.0, .1, .2, .25,  .3] :

    sys.stdout = open("./out"+str(learning_eps)+".txt", "a")

    big_print("train ADVERSARIAL ")
    mlp_yaml = open('mlp2.yaml', 'r').read()
    mlp_hyper_params = {'learning_eps': learning_eps, 'save_path' : '.'}
    mlp_yaml = mlp_yaml % mlp_hyper_params
    train_obj = yaml_parse.load(mlp_yaml)
    train_obj.main_loop()


