# Group 33, Machine learning project 1

You can obtain the same prediction as us by running the `run.py` file. 
Before that, you should first update some constant in this file to make
the script works.

You should edit the following constant :

* `DATA_TRAIN_PATH` : path to the train data csv
* `DATA_TEST_PATH` : path to the test data csv
* `OUTPUT_PATH` :  path where to store the output prediction csv

Also, you can observe the method we used – `logistic_regression` – and
the parameters we set for the model : *gamma*, *lambda*, etc.

Also, you can find our **cross-validation** code in 
`run.crossvalidation.py`. Where you can change the model used, and all
same kind of parameters and obtain a train and test score on this
execution.

* `models.py` contains our preprocessing fonctions and come with 
`utils.py` which contains utils functions
* `prepare_data.py` contains a `prepare_data` function which assembles
our preprocessing pipeline and output progress information
* `cross_validation.py` contains the `cross_validation` function 
where all model can be executed.