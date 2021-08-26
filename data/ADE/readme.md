Dataset for ADE. 10-fold data files can be found in the "/raw/" directory. 

For each fold, move "ade_split_train.json" and "ade_split_test.json" to the current directory 
and rename it as "train_triples.json" and "test_triples.json" before training. 

Further spliting of dev data from train data can be found in "/dataloader/dataloader.py" and is automatically done before training. 

The original files is retrived from (https://github.com/lavis-nlp/spert/tree/master/scripts). 
