import torch
import torch.nn as nn

class micro():
    def __init__(self, rel2idx, ner2idx):
        self.rel2idx = rel2idx
        self.ner2idx = ner2idx


    def get_right_entity_pair(self, ner_pred, ner_label):
        # ner_label : seq_len * seq_len * batch_size * entity_type

        ret = ner_label * ner_pred
        ret = torch.sum(ret, dim=1)
        ret = torch.sum(ret, dim=-1)
        ret = torch.where(ret > 0, torch.ones_like(ret), torch.zeros_like(ret))

        seq_len = ner_label.size(0)
        e1 = ret.unsqueeze(0).repeat(seq_len, 1, 1)
        e2 = ret.unsqueeze(1).repeat(1, seq_len, 1)
        ret = e1 * e2
        return ret

    def get_trip_pred(self, ner_pred, re_pred):

        ner_mask = torch.sum(ner_pred, dim=1).float()
        ner_mask = torch.sum(ner_mask, dim=-1).float()
        ner_mask = torch.where(ner_mask > 0, torch.ones_like(ner_mask), torch.zeros_like(ner_mask))

        seq_len = ner_mask.size(0)
        e1 = ner_mask.unsqueeze(0).repeat(seq_len, 1, 1)
        e2 = ner_mask.unsqueeze(1).repeat(1, seq_len, 1)
        ner_mask = e1 * e2

        ner_mask = ner_mask.unsqueeze(-1).repeat(1, 1, 1, len(self.rel2idx))
        complete_re_pred = re_pred * ner_mask
        return complete_re_pred

    def count_num(self, ner_pred, ner_label, re_pred, re_label):
        ner_pred = torch.where(ner_pred>=0.5, torch.ones_like(ner_pred),
                                    torch.zeros_like(ner_pred))
        re_pred = torch.where(re_pred>=0.5, torch.ones_like(re_pred),
                                    torch.zeros_like(re_pred))
        gold_num = re_label.sum().item()

        re_pred = self.get_trip_pred(ner_pred, re_pred)
        pred_num = re_pred.sum().item()

        re_right = re_pred + re_label
        re_right = torch.where(re_right == 2, torch.ones_like(re_right), torch.zeros_like(re_right))

        ner_right_mask = self.get_right_entity_pair(ner_pred, ner_label)
        ner_right_mask = ner_right_mask.unsqueeze(-1).repeat(1, 1, 1, re_label.size(-1))
        re_right = re_right * ner_right_mask
        right_num = re_right.sum().item()
        return [pred_num, gold_num, right_num]


    def count_ner_num(self, ner_pred, ner_label):
        ner_pred = torch.where(ner_pred>=0.5, torch.ones_like(ner_pred),
                                    torch.zeros_like(ner_pred))
        ner_pred_num = ner_pred.sum().item()
        ner_gold_num = ner_label.sum().item()

        ner_right = ner_pred * ner_label
        ner_right_num = ner_right.sum().item()
        return [ner_pred_num, ner_gold_num, ner_right_num]


class macro():
    def __init__(self, rel2idx, ner2idx):
        self.rel2idx = rel2idx
        self.ner2idx = ner2idx

    def get_right_entity_pair(self, ner_pred, ner_label):
        # ner_label : seq_len * seq_len * batch_size * entity_type

        ret = ner_label * ner_pred
        ret = torch.sum(ret, dim=1)
        ret = torch.sum(ret, dim=-1)
        ret = torch.where(ret > 0, torch.ones_like(ret), torch.zeros_like(ret))

        seq_len = ner_label.size(0)
        e1 = ret.unsqueeze(0).repeat(seq_len, 1, 1)
        e2 = ret.unsqueeze(1).repeat(1, seq_len, 1)
        ret = e1 * e2
        return ret

    def get_trip_pred(self, ner_pred, re_pred):

        ner_mask = torch.sum(ner_pred, dim=1).float()
        ner_mask = torch.sum(ner_mask, dim=-1).float()
        ner_mask = torch.where(ner_mask > 0, torch.ones_like(ner_mask), torch.zeros_like(ner_mask))

        seq_len = ner_mask.size(0)
        e1 = ner_mask.unsqueeze(0).repeat(seq_len, 1, 1)
        e2 = ner_mask.unsqueeze(1).repeat(1, seq_len, 1)
        ner_mask = e1 * e2

        complete_re_pred = re_pred * ner_mask
        return complete_re_pred

    def count_num(self, ner_pred, ner_label, re_pred, re_label):
        ner_pred = torch.where(ner_pred>=0.5, torch.ones_like(ner_pred),
                                    torch.zeros_like(ner_pred))
        re_pred = torch.where(re_pred>=0.5, torch.ones_like(re_pred),
                                    torch.zeros_like(re_pred))
        triple_num_list = []
        for i in range(len(self.rel2idx)):
            re_label_single = re_label[:, :, :, i]
            gold_num = re_label_single.sum().item()

            re_pred_single = self.get_trip_pred(ner_pred, re_pred[:, :, :, i])
            pred_num = re_pred_single.sum().item()

            re_right = re_pred_single + re_label_single
            re_right = torch.where(re_right == 2, torch.ones_like(re_right), torch.zeros_like(re_right))

            ner_right_mask = self.get_right_entity_pair(ner_pred, ner_label)
            re_right = re_right * ner_right_mask
            right_num = re_right.sum().item()
            triple_num_list += [pred_num, gold_num, right_num]

        return triple_num_list


    def count_ner_num(self, ner_pred, ner_label):
        ner_pred = torch.where(ner_pred>=0.5, torch.ones_like(ner_pred),
                                    torch.zeros_like(ner_pred))
        entity_num_list = []
        for i in range(len(self.ner2idx)):
            ner_pred_single = ner_pred[:, :, :, i]
            ner_label_single = ner_label[:, :, :, i]

            ner_pred_num = ner_pred_single.sum().item()
            ner_gold_num = ner_label_single.sum().item()

            ner_right = ner_pred_single * ner_label_single
            ner_right_num = ner_right.sum().item()
            entity_num_list += [ner_pred_num, ner_gold_num, ner_right_num]

        return entity_num_list


def f1(num):
    results = {}
    results["p"], results["r"], results["f"] = 0, 0, 0
    type_num = len(num)/3

    for i in range(0, len(num), 3):
        pred_num, gold_num, right_num = num[i], num[i+1], num[i+2]
        if pred_num == 0:
            p = 0
        else:
            p = float(right_num) / pred_num
        if gold_num == 0:
            r = 0
        else:
            r = float(right_num) / gold_num
        if p + r == 0:
            F1 = 0
        else:
            F1 = 2 * p * r / (p + r)


        results["p"] += p
        results["r"] += r
        results["f"] += F1
    results["p"] = results["p"] / type_num
    results["r"] = results["r"] / type_num
    results["f"] = results["f"] / type_num

    return results

class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()
        self.loss_ner = nn.BCELoss(reduction='sum')
        self.loss_re = nn.BCELoss(reduction='sum')

    def forward(self, ner_pred, ner_label, re_pred, re_label):
        seq_len = ner_pred.size(1)
        ner_loss = self.loss_ner(ner_pred, ner_label) / seq_len
        re_loss = self.loss_re(re_pred, re_label) / seq_len
        loss = ner_loss + re_loss
        return loss




