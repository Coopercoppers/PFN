## Difference between PFN and PFN-nested
In PFN-nested, we use two RE units for head-to-head and tail-to-tail prediction. There is only one RE unit in PFN model and it's for head-to-head prediction.

## Design Choice
The head-to-head RE unit takes relation feature as the main feature, shared feature as auxiliary feature.

The tail-to-tail RE unit takes shared feature as the main feature, relation feature as auxiliary feature.

## Justification
Shared feature act as a mutual section for NER and RE. 

By **sharing** NER information (head-to-tail) and head-to-head RE information, we should be able to get the tail-to-tail RE information.

This sharing mechanism is achieved by using shared feature as the main feature in the tail-to-tail RE unit.
