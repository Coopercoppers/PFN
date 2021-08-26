import json
import sys
import os
import argparse
from model.pfn import *
import torch
from transformers import AlbertTokenizer, AutoTokenizer
import re


def map_origin_word_to_bert(words, tokenizer):
    bep_dict = {}
    current_idx = 1

    for word_idx, word in enumerate(words):
        bert_word = tokenizer.tokenize(word)
        word_len = len(bert_word)
        bep_dict[word_idx] = [current_idx, current_idx + word_len - 1]
        current_idx = current_idx + word_len


    return bep_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument("--sent", type=str, required=True,
                        help="input sentence")
    parser.add_argument("--model_file", type=str, required=True,
                        help="loading pre-trained model files")
    parser.add_argument("--embed_mode", type=str,
                        help="loading pre-trained model files")
    parser.add_argument("--hidden_size", type=int, default=300,
                        help="hidden size of the model")
    parser.add_argument("--dropconnect", type=float, default=0.,
                        help="dropconnect on encoder")
    parser.add_argument("--dropout", type=float, default=0.,
                        help="dropout on word embedding and task units")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = None

    if "web" in args.model_file:
        data = "WEBNLG"
    elif "nyt" in args.model_file:
        data = "NYT"
    elif "ade" in args.model_file:
        data = "ADE"
    elif "ace" in args.model_file:
        data = "ACE2005"
    elif "sci" in args.model_file:
        data = "SCIERC"

    if "albert" in args.model_file:
        input_size = 4096
        args.embed_mode = "albert"
    elif "sci" in args.model_file:
        input_size = 768
        args.embed_mode = "scibert"
    elif "bert" in args.model_file:
        input_size = 768
        args.embed_mode = "bert_cased"



    with open("data/" + data + "/ner2idx.json", "r") as f:
        ner2idx = json.load(f)
    with open("data/" + data + "/rel2idx.json", "r") as f:
        rel2idx = json.load(f)

    idx2ner = {v: k for k, v in ner2idx.items()}
    idx2rel = {v: k for k, v in rel2idx.items()}

    model = PFN(args, input_size, ner2idx, rel2idx)
    model.load_state_dict(torch.load(args.model_file))
    model.to(device)
    model.eval()

    if args.embed_mode == "albert":
        tokenizer = AlbertTokenizer.from_pretrained("albert-xxlarge-v1")
    elif args.embed_mode == "bert_cased":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    else:
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

    target_sent = re.findall(r"\w+|[^\w\s]", args.sent)

    sent_bert_ids = tokenizer(target_sent, return_tensors="pt", is_split_into_words=True)["input_ids"].tolist()
    sent_bert_ids = sent_bert_ids[0]
    sent_bert_str = []
    for i in sent_bert_ids:
        sent_bert_str.append(tokenizer.convert_ids_to_tokens(i))


    bert_len = len(sent_bert_str)

    mask = torch.ones(bert_len, 1).to(device)
    ner_score, re_score = model(target_sent, mask)

    ner_score = torch.where(ner_score>=0.5, torch.ones_like(ner_score), torch.zeros_like(ner_score))
    re_score = torch.where(re_score>=0.5, torch.ones_like(re_score), torch.zeros_like(re_score))

    entity = (ner_score == 1).nonzero(as_tuple=False).tolist()
    relation = (re_score == 1).nonzero(as_tuple=False).tolist()

    word_to_bep = map_origin_word_to_bert(target_sent, tokenizer)
    bep_to_word = {word_to_bep[i][0]:i for i in word_to_bep.keys()}

    entity_names = {}
    for en in entity:
        type = idx2ner[en[3]]
        start = None
        end = None
        if en[0] in bep_to_word.keys():
            start = bep_to_word[en[0]]
        if en[1] in bep_to_word.keys():
            end = bep_to_word[en[1]]
        if start == None or end == None:
            continue

        entity_str = " ".join(target_sent[start:end+1])
        entity_names[entity_str] = start
        print("entity_name: {}, entity type: {}".format(entity_str, type))

    for re in relation:
        type = idx2rel[re[3]]

        e1 = None
        e2 = None

        if re[0] in bep_to_word.keys():
            e1 = bep_to_word[re[0]]
        if re[1] in bep_to_word.keys():
            e2 = bep_to_word[re[1]]
        if e1 == None or e2 == None:
            continue

        subj = None
        obj = None

        for en, start_index in entity_names.items():
            if en.startswith(target_sent[e1]) and start_index == e1:
                subj = en
            if en.startswith(target_sent[e2]) and start_index == e2:
                obj = en

        if subj == None or obj == None:
            continue

        print("triple: {}, {}, {}".format(subj, type, obj))











