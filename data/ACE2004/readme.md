Dataset for ACE2004 can be obtained from (https://catalog.ldc.upenn.edu/LDC2005T09). Note that the files are not for free!

After obtaining the files, preprocessing can be done by following the instructions from (https://github.com/LorrinWWW/two-are-better-than-one/tree/master/datasets)

Then, you should obtain two folder named "train" and "test", each contains 5-fold json files. For each fold, move {i}.json of ./train and ./test to the current folder, and rename them as "train_triples.json" and "test_triples.json".  

Further spliting of dev data from train data can be found in "/dataloader/dataloader.py" and is automatically done before training.


