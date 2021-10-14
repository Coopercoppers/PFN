import torch
import torch.nn as nn

class micro():
    def __init__(self, rel2idx, ner2idx):
        self.rel2idx = rel2idx
        self.ner2idx = ner2idx


    def get_ner_index(self, tensor):
        index = (tensor == 1).nonzero(as_tuple=False)
        index_scalar = []
        for index_tup in index:
            scalar = []
            for i in index_tup:
                scalar.append(i.item())
            index_scalar.append(tuple(scalar))
        return index_scalar

    def get_re_index(self, tensor):
        index = (tensor == 1).nonzero(as_tuple=False)
        index_list = []
        for index_tup in index:
            for i in index_tup:
                index_list.append(i.item())
        return index_list

    def get_trip(self, ner_pred, re_head_pred, re_tail_pred):
        seq_len = ner_pred.size(0)
        relation = len(self.rel2idx)


        re_head_pred = re_head_pred.view(seq_len * seq_len, relation)
        re_tail_pred = re_tail_pred.view(seq_len * seq_len, relation)

        ner_pred = torch.sum(ner_pred, dim=-1)
        ner_pred = torch.where(ner_pred > 0, torch.ones_like(ner_pred), torch.zeros_like(ner_pred))

        ner_pred_index = self.get_ner_index(ner_pred)
        ner_map = {}  # head to [(head,tail1),(head,tail2)]
        for tup in ner_pred_index:
            if tup[0] not in ner_map:
                ner_map[tup[0]] = [tup]
            else:
                ner_map[tup[0]].append(tup)


        full_trip = []

        for r in range(relation):
            re_head_pred_index = self.get_re_index(re_head_pred[:, r])
            re_tail_pred_index = self.get_re_index(re_tail_pred[:, r])

            for i in range(seq_len*seq_len):
                if i in re_head_pred_index:
                    subj_head = int(i // seq_len)
                    obj_head = int(i % seq_len)
                    if subj_head not in ner_map.keys() or obj_head not in ner_map.keys():
                        continue

                    subjects = ner_map[subj_head]
                    objects = ner_map[obj_head]

                    for s in subjects:
                        for o in objects:
                            posit = s[1] * seq_len + o[1]
                            if posit in re_tail_pred_index:
                                full_trip.append([s, r, o])

        return full_trip



    def count_num(self, ner_pred, ner_label, re_pred_head, re_pred_tail, re_label_head, re_label_tail):
        ner_pred = torch.where(ner_pred>=0.5, torch.ones_like(ner_pred),
                                    torch.zeros_like(ner_pred))
        re_pred_head = torch.where(re_pred_head>=0.5, torch.ones_like(re_pred_head),
                                    torch.zeros_like(re_pred_head))
        re_pred_tail = torch.where(re_pred_tail>=0.5, torch.ones_like(re_pred_tail),
                                    torch.zeros_like(re_pred_tail))


        batch = ner_pred.size(2)
        pred_num, gold_num, right_num = 0, 0, 0
        for i in range(batch):
            ner_pred_batch = ner_pred[:, :, i, :]
            ner_label_batch = ner_label[:, :, i, :]

            re_label_head_batch = re_label_head[:,:,i,:]
            re_label_tail_batch = re_label_tail[:,:,i,:]
            re_label_set = self.get_trip(ner_label_batch, re_label_head_batch, re_label_tail_batch)

            re_pred_head_batch = re_pred_head[:,:,i,:]
            re_pred_tail_batch = re_pred_tail[:,:,i,:]
            re_pred_set = self.get_trip(ner_pred_batch, re_pred_head_batch, re_pred_tail_batch)


            pred_num += len(re_pred_set)
            gold_num += len(re_label_set)

            re_right = [trip for trip in re_pred_set if trip in re_label_set]

            ner_right_batch = ner_pred_batch * ner_label_batch
            ner_right_batch = torch.sum(ner_right_batch, dim=-1)

            for trip in re_right:
                subject = trip[0]
                object = trip[2]
                if ner_right_batch[subject[0], subject[1]] > 0 and ner_right_batch[object[0], object[1]] > 0:
                    right_num += 1
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


    def get_ner_index(self, tensor):
        index = (tensor == 1).nonzero(as_tuple=False)
        index_scalar = []
        for index_tup in index:
            scalar = []
            for i in index_tup:
                scalar.append(i.item())
            index_scalar.append(tuple(scalar))
        return index_scalar

    def get_re_index(self, tensor):
        index = (tensor == 1).nonzero(as_tuple=False)
        index_list = []
        for index_tup in index:
            for i in index_tup:
                index_list.append(i.item())
        return index_list

    def get_trip(self, ner_pred, re_head_pred, re_tail_pred, relation):
        seq_len = ner_pred.size(0)


        re_head_pred = re_head_pred.view(seq_len * seq_len)
        re_tail_pred = re_tail_pred.view(seq_len * seq_len)

        ner_pred = torch.sum(ner_pred, dim=-1)
        ner_pred = torch.where(ner_pred > 0, torch.ones_like(ner_pred), torch.zeros_like(ner_pred))

        ner_pred_index = self.get_ner_index(ner_pred)
        ner_map = {}  # head to [(head,tail1),(head,tail2)]
        for tup in ner_pred_index:
            if tup[0] not in ner_map:
                ner_map[tup[0]] = [tup]
            else:
                ner_map[tup[0]].append(tup)


        full_trip = []


        re_head_pred_index = self.get_re_index(re_head_pred)
        re_tail_pred_index = self.get_re_index(re_tail_pred)

        for i in range(seq_len*seq_len):
            if i in re_head_pred_index:
                subj_head = int(i // seq_len)
                obj_head = int(i % seq_len)
                if subj_head not in ner_map.keys() or obj_head not in ner_map.keys():
                    continue

                subjects = ner_map[subj_head]
                objects = ner_map[obj_head]

                for s in subjects:
                    for o in objects:
                        posit = s[1] * seq_len + o[1]
                        if posit in re_tail_pred_index:
                            full_trip.append([s, relation, o])

        return full_trip


    def count_num(self, ner_pred, ner_label, re_pred_head, re_pred_tail, re_label_head, re_label_tail):
        ner_pred = torch.where(ner_pred>=0.5, torch.ones_like(ner_pred),
                                    torch.zeros_like(ner_pred))
        re_pred_head = torch.where(re_pred_head>=0.5, torch.ones_like(re_pred_head),
                                    torch.zeros_like(re_pred_head))
        re_pred_tail = torch.where(re_pred_tail>=0.5, torch.ones_like(re_pred_tail),
                                    torch.zeros_like(re_pred_tail))
        triple_num_list = []

        batch = ner_pred.size(2)
        for r in range(len(self.rel2idx)):
            pred_num, gold_num, right_num = 0, 0, 0
            for i in range(batch):
                ner_pred_batch = ner_pred[:, :, i, :]
                ner_label_batch = ner_label[:, :, i, :]

                re_label_head_batch = re_label_head[:,:,i,r]
                re_label_tail_batch = re_label_tail[:,:,i,r]
                re_label_set = self.get_trip(ner_label_batch, re_label_head_batch, re_label_tail_batch, r)

                re_pred_head_batch = re_pred_head[:,:,i,r]
                re_pred_tail_batch = re_pred_tail[:,:,i,r]
                re_pred_set = self.get_trip(ner_pred_batch, re_pred_head_batch, re_pred_tail_batch, r)


                pred_num += len(re_pred_set)
                gold_num += len(re_label_set)

                re_right = [trip for trip in re_pred_set if trip in re_label_set]

                ner_right_batch = ner_pred_batch * ner_label_batch
                ner_right_batch = torch.sum(ner_right_batch, dim=-1)

                for trip in re_right:
                    subject = trip[0]
                    object = trip[2]
                    if ner_right_batch[subject[0], subject[1]] > 0 and ner_right_batch[object[0], object[1]] > 0:
                        right_num += 1

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
        self.loss_re_head = nn.BCELoss(reduction='sum')
        self.loss_re_tail = nn.BCELoss(reduction='sum')


    def forward(self, ner_pred, ner_label, re_pred_head, re_pred_tail, re_label_head, re_label_tail):
        seq_len = ner_pred.size(1)
        ner_loss = self.loss_ner(ner_pred, ner_label) / seq_len
        re_head_loss = self.loss_re_head(re_pred_head, re_label_head) / seq_len
        re_tail_loss = self.loss_re_tail(re_pred_tail, re_label_tail) / seq_len
        loss = ner_loss + re_head_loss + re_tail_loss

        return loss




