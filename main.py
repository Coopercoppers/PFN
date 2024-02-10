import random
import json
import logging
import sys
import os
import torch
import argparse
import pickle
import numpy as np
from utils.metrics import *
from utils.helper import *
from model.pfn import PFN
from dataloader.dataloader import dataloader
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset,DataLoader
from helpers_al import VAE, query_samples, Discriminator


def train(args, model, train_batch, optimizer, BCEloss, dev_batch, rel2idx, ner2idx, test_batch):
    for epoch in range(args.epoch):#1
        steps, train_loss = 0, 0

        model.train()
        for data in tqdm(train_batch):
            steps+=1
            optimizer.zero_grad()

            text = data[0]
            ner_label = data[1].to(device)
            re_label = data[2].to(device)
            mask = data[-1].to(device)

            ner_pred, re_pred, features,_ = model(text, mask)
            loss = BCEloss(ner_pred, ner_label, re_pred, re_label)

            loss.backward()

            train_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.clip)
            optimizer.step()

            if steps % args.steps == 0:
                logger.info("Epoch: {}, step: {} / {}, loss = {:.4f}".format
                            (epoch, steps, len(train_batch), train_loss / steps))

        logger.info("------ Training Set Results ------")
        logger.info("loss : {:.4f}".format(train_loss / steps))

        if args.do_eval:
            model.eval()
            logger.info("------ Testing ------")
            dev_triple, dev_entity, dev_loss = evaluate(dev_batch, rel2idx, ner2idx, args, "dev", model)
            test_triple, test_entity, test_loss = evaluate(test_batch, rel2idx, ner2idx, args, "test", model)
            average_f1 = dev_triple["f"] + dev_entity["f"]

            if epoch == 0 or average_f1 > best_result:
                best_result = average_f1
                triple_best = test_triple
                entity_best = test_entity
                torch.save(model.state_dict(), output_dir + "/" + model_file)
                logger.info("Best result on dev saved!!!")


            saved_file.save("{} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}".format(epoch, train_loss/steps, dev_loss, test_loss, dev_entity["f"],
                                            dev_triple["f"], test_entity["f"], test_triple["f"]))

    saved_file.save("best test result ner-p: {:.4f} \t ner-r: {:.4f} \t ner-f: {:.4f} \t re-p: {:.4f} \t re-r: {:.4f} \t re-f: {:.4f} ".format(entity_best["p"],
                    entity_best["r"], entity_best["f"], triple_best["p"], triple_best["r"], triple_best["f"]))

    return model


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

        
class LossNet(nn.Module):
    def __init__(self, feature_sizes=[32, 16, 8, 4], num_channels=[64, 128, 256, 512], interm_dim=128):
        super(LossNet, self).__init__()
        
        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.FC1 = nn.Linear(num_channels[0]*9, interm_dim)
        self.FC2 = nn.Linear(num_channels[1]*9, interm_dim)
        self.FC3 = nn.Linear(num_channels[2]*9, interm_dim)
        self.FC4 = nn.Linear(num_channels[3]*9, interm_dim)

        self.linear = nn.Linear(4 * interm_dim, 1)
    
    def forward(self, features):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = self.GAP4(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))

        out = self.linear(torch.cat((out1, out2, out3, out4), 1))
        return out
    

    
class BasicBlock(nn.Module):
    
    def __init__(self, in_planes, planes, expansion, stride=1):
        super(BasicBlock, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(test_batch, rel2idx, ner2idx, args, test_or_dev, model):
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

    with torch.no_grad():
        for data in test_batch:
            steps += 1
            text = data[0]
            ner_label = data[1].to(device)
            re_label = data[2].to(device)
            mask = data[-1].to(device)

            ner_pred, re_pred, features,_ = model(text, mask)
            loss = BCEloss(ner_pred, ner_label, re_pred, re_label)
            test_loss += loss

            entity_num = metric.count_ner_num(ner_pred, ner_label)
            triple_num = metric.count_num(ner_pred, ner_label, re_pred, re_label)

            for i in range(len(entity_num)):
                total_entity_num[i] += entity_num[i]
            for i in range(len(triple_num)):
                total_triple_num[i] += triple_num[i]


        triple_result = f1(total_triple_num)
        entity_result = f1(total_entity_num)

        logger.info("------ {} Results ------".format(test_or_dev))
        logger.info("loss : {:.4f}".format(test_loss / steps))
        logger.info("entity: p={:.4f}, r={:.4f}, f={:.4f}".format(entity_result["p"], entity_result["r"], entity_result["f"]))
        logger.info("triple: p={:.4f}, r={:.4f}, f={:.4f}".format(triple_result["p"], triple_result["r"], triple_result["f"]))

    return triple_result, entity_result, test_loss / steps
  
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, type=str, required=True,
                        help="which dataset to use")

    parser.add_argument("--epoch", default=100, type=int,
                        help="number of training epoch")

    parser.add_argument("--hidden_size", default=300, type=int,
                        help="number of hidden neurons in the model")

    parser.add_argument("--batch_size", default=20, type=int,
                        help="number of samples in one training batch")
    
    parser.add_argument("--eval_batch_size", default=10, type=int,
                        help="number of samples in one testing batch")
    
    parser.add_argument("--do_train", action="store_true",
                        help="whether or not to train from scratch")

    parser.add_argument("--do_eval", action="store_true",
                        help="whether or not to evaluate the model")

    parser.add_argument("--embed_mode", default=None, type=str, required=True,
                        help="BERT or ALBERT pretrained embedding")

    parser.add_argument("--eval_metric", default="micro", type=str,
                        help="micro f1 or macro f1")

    parser.add_argument("--lr", default=None, type=float,
                        help="initial learning rate")

    parser.add_argument("--weight_decay", default=0, type=float,
                        help="weight decaying rate")
    
    parser.add_argument("--linear_warmup_rate", default=0.0, type=float,
                        help="warmup at the start of training")
    
    parser.add_argument("--seed", default=0, type=int,
                        help="random seed initiation")

    parser.add_argument("--dropout", default=0.1, type=float,
                        help="dropout rate for input word embedding")

    parser.add_argument("--dropconnect", default=0.1, type=float,
                        help="dropconnect rate for partition filter layer")

    parser.add_argument("--steps", default=50, type=int,
                        help="show result for every 50 steps")

    parser.add_argument("--output_file", default="test", type=str, required=True,
                        help="name of result file")

    parser.add_argument("--clip", default=0.25, type=float,
                        help="grad norm clipping to avoid gradient explosion")

    parser.add_argument("--max_seq_len", default=128, type=int,
                        help="maximum length of sequence")


    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    output_dir = args.output_file
    os.mkdir(output_dir)

    logger.addHandler(logging.FileHandler(output_dir + "/" + args.output_file + ".log", 'w'))
    logger.info(sys.argv)
    logger.info(args)

    saved_file = save_results(output_dir + "/" + args.output_file + ".txt", header="# epoch \t train_loss \t  dev_loss \t test_loss \t dev_ner \t dev_rel \t test_ner \t test_rel")
    model_file = args.output_file + ".pt"
       
    with open(args.data + "/ner2idx.json", "r") as f:
        ner2idx = json.load(f)
    with open(args.data + "/rel2idx.json", "r") as f:
        rel2idx = json.load(f)

    train_dataset, test_dataset, dev_dataset, collate_fn, train_unlabeled = dataloader(args, ner2idx, rel2idx)

    adden = 50 # roughly no_train/cycles
    no_train = len(train_dataset)
    ADDENDUM = adden
    NUM_TRAIN = no_train
    indices = list(range(NUM_TRAIN))
    random.shuffle(indices)

    labeled_set = indices[:ADDENDUM]
    unlabeled_set = [x for x in indices if x not in labeled_set]

    train_batch = DataLoader(dataset=train_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(labeled_set), 
                                pin_memory=True, collate_fn=collate_fn)
    test_batch = DataLoader(dataset=test_dataset, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn)

    dev_batch = DataLoader(dataset=dev_dataset, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn)

    
    vae = VAE()
    if args.embed_mode == "albert":
            input_size = 4096
    else:
        input_size = 768

    model = PFN(args, input_size, ner2idx, rel2idx)
    discriminator = Discriminator(32)
    loss_module = LossNet().to(device)
    # models = {'backbone': model['backbone'], 'module': model['module'], 'vae': vae, 'discriminator': discriminator}
    models      = {'backbone': model, 'module': loss_module, 'vae': vae, 'discriminator': discriminator}
    
    weight1 = nn.Parameter(torch.ones(1))
    weight2 = nn.Parameter(torch.ones(1))
    weight1 = weight1.to(device)
    weight2 = weight2.to(device)
    weights = [weight1, weight2]
    vae = models['vae']
    # discriminator = models['discriminator']
    # task_model = models['backbone']
    ranker = models['module']

    if args.do_train:
        logger.info("------Training------")
        if args.embed_mode == "albert":
            input_size = 4096
        else:
            input_size = 768

        
        model.to(device)


        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if args.eval_metric == "micro":
            metric = micro(rel2idx, ner2idx)
        else:
            metric = macro(rel2idx, ner2idx)

        BCEloss = loss()
        best_result = 0
        triple_best = None
        entity_best = None
        method = 'TA-VAAL'
        for cycle in range(7):
            print(cycle)
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:50]

            models['backbone'] = train(args, models['backbone'], train_batch, optimizer, BCEloss, dev_batch, rel2idx, ner2idx, test_batch)
            torch.save(models['backbone'], 'predictor-backbone-' + 'cycle-'+str(cycle+1)+'.pth')
            torch.save(models['module'], 'predictor-module-'+'cycle-'+str(cycle+1)+'.pth')
            arg = query_samples(models, method, train_unlabeled, subset, labeled_set, cycle, args,collate_fn,weights)

            new_list = list(torch.tensor(subset)[arg][:30].numpy())
            labeled_set += list(torch.tensor(subset)[arg][-30:].numpy())
            listd = list(torch.tensor(subset)[arg][:-30].numpy()) 
            unlabeled_set = listd + unlabeled_set[50:]
            print(len(labeled_set), min(labeled_set), max(labeled_set))
            
            np.save("labelled-" + 'head' + str(cycle) + ".npy", np.array(labeled_set))
            #saved_history/
            # Create a new dataloader for the updated labeled dataset
            train_batch = DataLoader(dataset=train_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(labeled_set), 
                                        pin_memory=True, collate_fn=collate_fn)

