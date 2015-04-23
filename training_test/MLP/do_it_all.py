from pylearn2.config import yaml_parse
import costAdv
import sys
# sys.stdout = open("./out.txt", "w")



def big_print(message):
	print ""
	print ""
	print "********************************************************"
	print ""
	print "==> " + message
	print ""
	print "********************************************************"
	print ""
	print ""

def adv_learning(learning_eps,training_eps):

    # Do some prints
    # sys.stdout = open("./out"+str(learning_eps)+"_"+str(training_eps)+".txt", "w")
    msg = "Compute adversarial dataset => " + str(training_eps)
    big_print(msg)
    
    # Compute adversarial dataset
    costAdv.make_adv_training_set(learning_eps, training_eps)

    # Test the model on adversarail dataset
    mlp_test_yaml = open('mlp_test.yaml', 'r').read()
    mlp_hyper_params = {'training_eps' : training_eps,
                        'learning_eps' : learning_eps, 
                        'save_path' : '.'}
    mlp_test_yaml = mlp_test_yaml % (mlp_hyper_params)
    train_obj = yaml_parse.load(mlp_test_yaml)
    train_obj.main_loop()




for learning_eps in [.0, .1, .2, .25,  .3] :

    sys.stdout = open("./out"+str(learning_eps)+".txt", "w")

    big_print("train ADVERSARIAL ")
    mlp_yaml = open('mlp.yaml', 'r').read()
    mlp_hyper_params = {'learning_eps' : learning_eps, 'save_path' : '.'}
    mlp_yaml = mlp_yaml % (mlp_hyper_params)
    train_obj = yaml_parse.load(mlp_yaml)
    train_obj.main_loop()

    
    adv_learning(learning_eps, .007)
    adv_learning(learning_eps, .1)
    adv_learning(learning_eps, .2)
    adv_learning(learning_eps, .25)
    adv_learning(learning_eps, .3)

