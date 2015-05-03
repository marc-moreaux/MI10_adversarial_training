To compte everything so far :

------------------------------------------------------
-- 1 -------------------------------------------------
------------------------------------------------------
Train the models mlp1 and mlp2 on MNIST 
dataset.
-> 1 - mlp1 is a (1200 + 10) SNN
-> 2 - mlp2 is a (1200 + 1200 + 10) SNN
$ python -u m_train.py



------------------------------------------------------
-- 2 -------------------------------------------------
------------------------------------------------------
Compute adversarial datasets and test 
the previous models on them. The datasets
can be :
-> advertially modified
-> Normmal noised modified
-> Uniform noise modified
$ python -u m_test.py


------------------------------------------------------
-- 3 -------------------------------------------------
------------------------------------------------------
Draw the result. This will call 
"Model_wrapper" wich all the models made 
so far and have functions to plot some 
results
$ python -u m_draw.py
