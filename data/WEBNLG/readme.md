Dataset for WEBNLG, the original files could be obtained from (https://github.com/weizhepei/CasRel/tree/master/data/WebNLG).  


The original files has 171 relation types and our files has 170 relation types. Only one sample belongs to the additional relation (located in **dev_triples.json**).

If you obtain the original files from the link above, please manually delete the following sample before training:  


{'text': 'The length of the first runway at Amsterdam Airport Schiphol is 3800 metres .', 'triple_list': [['Schiphol', '1st_runway_LengthMetre', '3800']]}
