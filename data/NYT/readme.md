Unzip the files, there should be at least three files: "train_triples.json", "dev_triples.json", "test_triples.json". The original files can be obtained from (https://github.com/weizhepei/CasRel/tree/master/data/NYT). 

We manually modified one sentence in **train_triples.json**:   
{"text": "XXX XXXXXX XXXXXXXXX XXXXXXXXXX XXXXXXXXXX XXXXXXXX XXXXXXXX XXXX XXXXXXX XXXXXXXXX XXXXXXXXX XXXXX XXXXXXXXXX XXXXXXXXXX XXXXXXXXXXX XXXX XXXXXXXXXX XXXXXXXXXXX XXXXXXXXX XXXXXXXXXXXX XXXX XXX XXXXXX XXXXXXX XXXXXX XXXXXXXXXX XXXXXXXXXXXXX XXXX XXXXXXXX XXXX XXXXXXXX XXXX XX XXXXX XXXXX XXXXX XXXXXXXX XXXX XXXXX XXXX XXXXX XXX XXXXXXX XXXXXXX XXXX XXXXX XXXXX XXXXXX XX XXXXX XXX XX XXXXXXX This demonstrated to Afghan warlords that they could not play America and Iran off one another and prompted Tehran to deport hundreds of suspected Al Qaeda and Taliban operatives who had fled Afghanistan ."} 

By removing all the X symbols to avoid out of memory error. If you obtain the files from the above link, please remove those XXX before training.

