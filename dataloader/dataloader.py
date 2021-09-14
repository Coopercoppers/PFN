import json
from utils.helper import *
from transformers import AlbertTokenizer, AutoTokenizer
from torch.utils.data import Dataset,DataLoader
import random

class dataprocess(Dataset):
    def __init__(self, data, embed_mode, max_seq_len):
        self.data = data
        self.len = max_seq_len
        if embed_mode == "albert":
            self.tokenizer = AlbertTokenizer.from_pretrained("albert-xxlarge-v1")
        elif embed_mode == "bert_cased":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        elif embed_mode == "scibert":
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words = self.data[idx][0]
        ner_labels = self.data[idx][1]
        rc_labels = self.data[idx][2]
        
        if len(words) > self.len:
            words, ner_labels, rc_labels = self.truncate(self.len, words, ner_labels, rc_labels)

        sent_str = ' '.join(words)
        bert_words = self.tokenizer.tokenize(sent_str)
        bert_len = len(bert_words) + 2
        # bert_len = original sentence + [CLS] and [SEP]

        word_to_bep = self.map_origin_word_to_bert(words)
        ner_labels = self.ner_label_transform(ner_labels, word_to_bep)
        rc_labels = self.rc_label_transform(rc_labels, word_to_bep)

        return (words, ner_labels, rc_labels, bert_len)

    def map_origin_word_to_bert(self, words):
        bep_dict = {}
        current_idx = 0
        for word_idx, word in enumerate(words):
            bert_word = self.tokenizer.tokenize(word)
            word_len = len(bert_word)
            bep_dict[word_idx] = [current_idx, current_idx + word_len - 1]
            current_idx = current_idx + word_len
        return bep_dict

    def ner_label_transform(self, ner_label, word_to_bert):
        new_ner_labels = []

        for i in range(0, len(ner_label), 3):
            # +1 for [CLS]
            sta = word_to_bert[ner_label[i]][0] + 1
            end = word_to_bert[ner_label[i + 1]][0] + 1
            new_ner_labels += [sta, end, ner_label[i + 2]]

        return new_ner_labels

    def rc_label_transform(self, rc_label, word_to_bert):
        new_rc_labels = []

        for i in range(0, len(rc_label), 3):
            # +1 for [CLS]
            e1 = word_to_bert[rc_label[i]][0] + 1
            e2 = word_to_bert[rc_label[i + 1]][0] + 1
            new_rc_labels += [e1, e2, rc_label[i + 2]]

        return new_rc_labels
    
    def truncate(self, max_seq_len, words, ner_labels, rc_labels):
        truncated_words = words[:max_seq_len]
        truncated_ner_labels = []
        truncated_rc_labels = []
        for i in range(0, len(ner_labels), 3):
            if ner_labels[i] < max_seq_len and ner_labels[i+1] < max_seq_len:
                truncated_ner_labels += [ner_labels[i], ner_labels[i+1], ner_labels[i+2]]

        for i in range(0, len(rc_labels), 3):
            if rc_labels[i] < max_seq_len and rc_labels[i+1] < max_seq_len:
                truncated_rc_labels += [rc_labels[i], rc_labels[i+1], rc_labels[i+2]]

        return truncated_words, truncated_ner_labels, truncated_rc_labels


def ace_preprocess(data):
    processed = []
    for dic in data:
        ner_tags = dic['ner']
        texts = dic['sentences']
        re_tags = dic['relations']

        cur_len = 0
        for idx, text in enumerate(texts):            
            ner_tag = ner_tags[idx]
            re_tag = re_tags[idx]
            ner_labels = []
            rc_labels = []
            for ner in ner_tag:
                ner[0] = ner[0] - cur_len
                ner[1] = ner[1] - cur_len
                ner_labels += ner
            for re in re_tag:
                new_re = []
                new_re.append(re[0] - cur_len)
                new_re.append(re[2] - cur_len)
                new_re.append(re[4])
                rc_labels += new_re

            processed += [(text, ner_labels, rc_labels)]
            cur_len += len(text)
    return processed

def nyt_and_webnlg_preprocess(data):
    processed = []
    for dic in data:
        text = dic['text']
        text = text.split(" ")
        ner_labels = []
        rc_labels = []
        trips = dic['triple_list']
        for trip in trips:
            subj = text.index(trip[0])
            obj = text.index(trip[2])
            rel = trip[1]
            if subj not in ner_labels:
                ner_labels +=[subj,subj,"None"]
            if obj not in ner_labels:
                ner_labels +=[obj,obj,"None"]

            rc_labels+=[subj,obj,rel]

        processed += [(text,ner_labels,rc_labels)]
    return processed


def ade_and_sci_preprocess(data, dataset):
    processed = []
    for dic in data:
        text = dic['tokens']
        ner_labels = []
        rc_labels = []
        entity = dic['entities']
        relation = dic['relations']

        for en in entity:
            ner_labels+=[en['start'], en['end']-1, en['type']]

        for re in relation:
            subj_idx = re['head']
            obj_idx = re['tail']
            subj = entity[subj_idx]
            obj = entity[obj_idx]
            rc_labels+=[subj['start'], obj['start'], re['type']]

        overlap_pattern = False
        if dataset == "ADE":
            for i in range(0, len(ner_labels), 3):
                for j in range(i+3, len(ner_labels), 3):
                    if is_overlap([ner_labels[i], ner_labels[i+1]], [ner_labels[j], ner_labels[j+1]]):
                        overlap_pattern = True
                        break
        if overlap_pattern == True:
            continue
        
        processed += [(text,ner_labels,rc_labels)]
    return processed


def dataloader(args, ner2idx, rel2idx):
    path = "data/" + args.data

    if args.data == "ADE":
        train_raw_data = json_load(path, "train_triples.json")
        test_data = json_load(path, "test_triples.json")
        random.shuffle(train_raw_data)
        split = int(0.15 * len(train_raw_data))
        train_data = train_raw_data[split:]
        dev_data = train_raw_data[:split]
    
    elif args.data == "ACE2004":
        train_raw_data = json_loads(path, "train_triples.json")
        test_data = json_loads(path, "test_triples.json")
        random.shuffle(train_raw_data)
        split = int(0.15 * len(train_raw_data))
        train_data = train_raw_data[split:]
        dev_data = train_raw_data[:split]
    
    elif args.data == "ACE2005":
        train_data = json_loads(path, 'train_triples.json')
        test_data = json_loads(path, 'test_triples.json')
        dev_data = json_loads(path, 'dev_triples.json')

    else:
        train_data = json_load(path, 'train_triples.json')
        test_data = json_load(path, 'test_triples.json')
        dev_data = json_load(path, 'dev_triples.json')



    if args.data=="ACE2005" or args.data=="ACE2004":
        train_data = ace_preprocess(train_data)
        test_data = ace_preprocess(test_data)
        dev_data = ace_preprocess(dev_data)


    elif args.data=="ADE" or args.data=="SCIERC" or args.data=="CONLL04":
        train_data = ade_and_sci_preprocess(train_data, args.data)
        test_data = ade_and_sci_preprocess(test_data, args.data)
        dev_data = ade_and_sci_preprocess(dev_data, args.data)

    else:
        train_data = nyt_and_webnlg_preprocess(train_data)
        test_data = nyt_and_webnlg_preprocess(test_data)
        dev_data = nyt_and_webnlg_preprocess(dev_data)


    train_dataset = dataprocess(train_data, args.embed_mode, args.max_seq_len)
    test_dataset = dataprocess(test_data, args.embed_mode, args.max_seq_len)
    dev_dataset = dataprocess(dev_data, args.embed_mode, args.max_seq_len)
    collate_fn = collater(ner2idx, rel2idx)


    train_batch = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn)
    test_batch = DataLoader(dataset=test_dataset, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn)
    dev_batch = DataLoader(dataset=dev_dataset, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn)


    return train_batch, test_batch, dev_batch
