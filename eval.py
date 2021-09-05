import json
import logging
import sys
import torch
import argparse
import pickle
import numpy as np
from utils.metrics import *
from utils.helper import *
from dataloader.dataloader import *
from tqdm import tqdm
from model.pfn import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, type=str, required=True,
                        help="which dataset to use")

    parser.add_argument("--batch_size", default=20, type=int,
                        help="number of samples in one training batch")
    
    parser.add_argument("--eval_batch_size", default=10, type=int,
                        help="number of samples in testing")

    parser.add_argument("--eval_metric", default="micro", type=str,
                        help="micro f1 or macro f1")

    parser.add_argument("--model_file", default="saved_model.pt", type=str, required=True,
                        help="file of saved model")

    parser.add_argument("--embed_mode", default=None, type=str, required=True,
                        help="BERT or ALBERT pretrained embedding")

    parser.add_argument("--hidden_size", default=300, type=int,
                        help="number of hidden neurons in the model")

    parser.add_argument("--dropout", default=0.1, type=float,
                        help="dropout rate for input word embedding")

    parser.add_argument("--dropconnect", default=0.1, type=float,
                        help="dropconnect rate for partition filter layer")
    
    parser.add_argument("--max_seq_len", default=128, type=int,
                        help="maximum length of sequence")


    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(sys.argv)
    logger.info(args)


    with open("data/" + args.data + "/ner2idx.json", "r") as f:
        ner2idx = json.load(f)
    with open("data/" + args.data + "/rel2idx.json", "r") as f:
        rel2idx = json.load(f)

    train_batch, test_batch, dev_batch = dataloader(args, ner2idx, rel2idx)


    if args.embed_mode == "albert":
        input_size = 4096
    else:
        input_size = 768

    model = PFN(args, input_size, ner2idx, rel2idx)
    model.load_state_dict(torch.load(args.model_file))
    model.to(device)
    model.eval()

    steps, test_loss = 0, 0
    total_triple_num = [0, 0, 0]
    total_entity_num = [0, 0, 0]
    if args.eval_metric == "macro":
        total_triple_num *= len(rel2idx)
        total_entity_num *= len(ner2idx)

    if args.eval_metric == "micro":
        metric = micro(rel2idx, ner2idx)
    else:
        metric = macro(rel2idx, ner2idx)

    logger.info("------ Testing ------")
    with torch.no_grad():
        for data in tqdm(test_batch):
            steps += 1
            text = data[0]
            ner_label = data[1].to(device)
            re_label = data[2].to(device)
            mask = data[-1].to(device)

            ner_pred, re_pred = model(text, mask)
            entity_num = metric.count_ner_num(ner_pred, ner_label)
            triple_num = metric.count_num(ner_pred, ner_label, re_pred, re_label)

            for i in range(len(entity_num)):
                total_entity_num[i] += entity_num[i]
            for i in range(len(triple_num)):
                total_triple_num[i] += triple_num[i]

        triple_result = f1(total_triple_num)
        entity_result = f1(total_entity_num)

        logger.info("------ Test Results ------")
        logger.info("entity: p={:.4f}, r={:.4f}, f={:.4f}".format(entity_result["p"], entity_result["r"], entity_result["f"]))
        logger.info("triple: p={:.4f}, r={:.4f}, f={:.4f}".format(triple_result["p"], triple_result["r"], triple_result["f"]))
