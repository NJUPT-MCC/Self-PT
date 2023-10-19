from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import random
from multiprocessing import Pool
import h5py
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
import re


from torch.utils.data.distributed import DistributedSampler


import transformers
from transformers import T5TokenizerFast
from tokenization import FewVLMTokenizerFast


project_dir = Path(__file__).resolve().parent.parent  # Self_PT/
workspace_dir = project_dir
dataset_dir = workspace_dir.joinpath('datasets/').resolve()
# coco_dir = dataset_dir.joinpath('VQA')
vg_dir = dataset_dir.joinpath('GQA')
# coco_img_dir = coco_dir.joinpath('images/')
# coco_feature_dir = coco_dir.joinpath('features')
gqa_dir = dataset_dir.joinpath('GQA')


class GQAFineTuneDataset(Dataset):
    def __init__(self, split='train,valid', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train'):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.mode = mode

        # Loading datasets to data
        self.sources = split.split(',')
        if self.verbose:
            print('Data sources: ', self.sources)

        if 't5' in self.args.backbone:
            if self.args.use_vision:
                self.tokenizer = FewVLMTokenizerFast.from_pretrained(
                    args.backbone,
                    # max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)
            else:
                self.tokenizer = T5TokenizerFast.from_pretrained(
                    args.backbone,
                    # max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)

        self.img_ids_to_source = {}
        data_info_dicts = []
        for source in self.sources:
            data_info_path = dataset_dir.joinpath(f'GQA/{source}.json')
            with open(data_info_path) as f:
                _data_info_dicts = json.load(f)
                # source_img_ids.append([d['img_id'] for d in _data_info_dicts])
                for _d in _data_info_dicts:
                    self.img_ids_to_source[_d['img_id']] = source
                    _d['source'] = source

                data_info_dicts.extend(_data_info_dicts)
            if self.verbose:
                print(f"Loaded {len(_data_info_dicts)} data from", source)

        data = data_info_dicts

        self.n_gpus = torch.cuda.device_count()

        self.rank = rank

        if self.topk > 0:
            data = data[:self.topk]
            if self.verbose:
                print(f"Use only {self.topk} data")

        self.data = data

        if args.subsample and split == 'train':
            random.seed(args.dataseed)
            random.shuffle(self.data)
            if 'train' in split and mode == 'train':
                self.data = self.data[:args.num_data]
            elif 'train' in split and mode == 'val':
                self.data = self.data[args.num_data:2*args.num_data]

        if self.verbose:
            print("# all sentences:", len(self.data))

        self.n_boxes = args.n_boxes

        self.source_to_featname = {
            'train': 'others',
            'valid': 'others',
            'submit': 'others',
            'train_10': 'others', 'train_20': 'others', 'train_30': 'others', 'train_40': 'others', 'train_50': 'others',
            'val_10': 'others', 'val_20': 'others', 'val_30': 'others', 'val_40': 'others', 'val_50': 'others', 
            'train_30_2': 'others', 'train_30_3': 'others',  'val_30_2': 'others', 'val_30_3': 'others',  
            'train_16': 'others', 'train_16_2': 'others', 'train_16_3': 'others',  'val_16': 'others', 'val_16_2': 'others', 'val_16_3': 'others', 
            'train_4': 'others', 'train_4_2': 'others', 'train_4_3': 'others',  'val_4': 'others', 'val_4_2': 'others', 'val_4_3': 'others',      
            'testdev': 'testdev'
        }

        self.featname_to_h5 = {
            'others': vg_dir.joinpath('vg_gqa_obj36.h5'),
            'testdev': gqa_dir.joinpath('gqa_testdev_obj36.h5'),
        }


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]
        # uid = datum['uid']
        # out_dict['uid'] = uid
        # out_dict['uid'] = uid

        ###### Image ######
        if self.args.use_vision:
            img_id = datum['img_id']
            out_dict['img_id'] = img_id

            source = self.img_ids_to_source[img_id]

            featname = self.source_to_featname[source]

            # f = self.source_to_h5[source]
            f = self.featname_to_h5[featname]

            if isinstance(f, Path):
                # path = self.data_source_to_h5_path[source]
                f = h5py.File(f, 'r')
                # self.split_to_h5_features[split_i] = f
                # self.source_to_h5[source] = f
                self.featname_to_h5[featname] = f

            feats = np.zeros(shape=(self.n_boxes, 2048), dtype=np.float32)
            f[f'{img_id}/features'].read_direct(feats)
            feats = torch.from_numpy(feats)
            out_dict['vis_feats'] = feats

            # Normalize the boxes (to 0 ~ 1)
            img_h = f[f'{img_id}/img_h'][()]
            img_w = f[f'{img_id}/img_w'][()]
            boxes = f[f'{img_id}/boxes'][()]  # (x1, y1, x2, y2)
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            np.testing.assert_array_less(boxes, 1+1e-5)
            # np.testing.assert_array_less(boxes, 1+5e-2)
            np.testing.assert_array_less(-boxes, 0+1e-5)
            boxes = torch.from_numpy(boxes)

            boxes.clamp_(min=0.0, max=1.0)

            out_dict['boxes'] = boxes

        ###### Text #####
        # caption = datum['caption']
        sent = datum['sent']

        input_ids = self.tokenizer.encode(f'{sent} <extra_id_0>', max_length=20, truncation=True)
        question_id = datum['question_id']
        out_dict['question_id'] = question_id
        out_dict['sent'] = sent
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)

        if 'label' in datum:
            label = datum['label']
            out_dict['label'] = label

            # https://github.com/airsplay/lxmert/blob/master/src/pretrain/lxmert_pretrain.py#L191
            answers = []
            scores = []
            for a, s in label.items():
                answers.append(a)
                scores.append(s)

            score_sum = sum(scores)

            if score_sum == 0:
                answer = ''
                score = 0.
            else:
                prob = [score / score_sum for score in scores]
                choice = np.random.multinomial(1, prob).argmax()
                answer = answers[choice]
                score = scores[choice]
                assert len(answer) > 0, (sent, label, choice, answer)

            out_dict['answer'] = answer
            out_dict['score'] = score
            out_dict['all_answers'] = answers

            if sum(scores) > 0:
                best_answers = []
                best_score = max(scores)
                for a, s in label.items():
                    if s == best_score and s > 0:
                        best_answers.append(a)
                out_dict['best_answers_tokenized'] = [self.tokenizer.encode(a) for a in best_answers]
            else:
                out_dict['best_answers_tokenized'] = [[]]
            

            target_ids = self.tokenizer.encode(f'<extra_id_0> {answer}')
            out_dict['target_ids'] = torch.LongTensor(target_ids)
            out_dict['target_length'] = len(target_ids)

        return out_dict


    def collate_fn(self, batch):
        batch_entry = {}

        args = batch[0]['args']

        B = len(batch)

        S_W_L = max(entry['input_length'] for entry in batch)
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        if args.use_vision:
            V_L = len(batch[0]['boxes'])
            feat_dim = batch[0]['vis_feats'].shape[-1]

            boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
            vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)

        if 'target' in batch[0]:
            # targets = []
            targets = torch.zeros(B, len(batch[0]['target']), dtype=torch.float)
        if 'target_ids' in batch[0]:
            T_W_L = max(entry['target_length'] for entry in batch)
            target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        sentences = []
        question_ids = []
        answers = []
        all_answers = []
        all_answers_tokenized = []
        best_answers_tokenized = []
        img_ids = []
        img_paths = []
        labels = []
        scores = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']

            if args.use_vision:
                boxes[i] += entry['boxes']
                vis_feats[i] += entry['vis_feats']
                # img_ids.append(entry['img_id'])
                # img_paths.append(entry['img_path'])

            if 'target_ids' in entry:
                target_ids[i, :entry['target_length']] = entry['target_ids']

            if 'target' in entry:
                targets[i] += entry['target']
                # targets.append(entry['target'])

            sentences.append(entry['sent'])
            question_ids.append(entry['question_id'])
            if 'answer' in entry:
                answers.append(entry['answer'])
            if 'all_answers' in entry:
                all_answers.append(entry['all_answers'])
            if 'all_answers_tokenized' in entry:
                all_answers_tokenized.append(entry['all_answers_tokenized'])
            if 'best_answers_tokenized' in entry:
                best_answers_tokenized.append(entry['best_answers_tokenized'])
            if 'score' in entry:
                scores.append(entry['score'])

            if 'label' in entry:
                labels.append(entry['label'])

        batch_entry['input_ids'] = input_ids
        if 'target_ids' in batch[0]:
            word_mask = target_ids != self.tokenizer.pad_token_id
            target_ids[~word_mask] = -100
            batch_entry['target_ids'] = target_ids
        if 'target' in batch[0]:
            # targets = torch.stack(targets, dim=0)
            batch_entry['targets'] = targets

        if args.use_vision:
            batch_entry['boxes'] = boxes
            batch_entry['vis_feats'] = vis_feats
            # batch_entry['img_id'] = img_ids
            # batch_entry['img_paths'] = img_paths

        batch_entry['sent'] = sentences
        batch_entry['question_ids'] = question_ids
        batch_entry['answers'] = answers
        batch_entry['all_answers'] = all_answers
        batch_entry['all_answers_tokenized'] = all_answers_tokenized
        batch_entry['best_answers_tokenized'] = best_answers_tokenized
        batch_entry['scores'] = torch.FloatTensor(scores)
        batch_entry['labels'] = labels

        batch_entry['task'] = 'gqa'

        return batch_entry


def get_loader(args, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1, verbose=None):

    if verbose is None:
        verbose = (gpu == 0)

    _dset = GQADataset(split, verbose)

    dataset = GQAFineTuneDataset(
        split,
        raw_dataset=_dset,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
        mode=mode)

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None
    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)

    loader.evaluator = GQAEvaluator(_dset)
    loader.task = 'gqa'

    return loader


class GQADataset:
    """
    A GQA data example in json file:
    {
        "img_id": "2375429",
        "label": {
            "pipe": 1.0
        },
        "question_id": "07333408",
        "sent": "What is on the white wall?"
    }
    """

    def __init__(self, splits: str, verbose=True):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets to data
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open(gqa_dir.joinpath("%s.json" % split))))
        if verbose:
            print("Load %d data from split(s) %s." %
                  (len(self.data), self.name))

        # List to dict (for evaluation and others)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open(gqa_dir.joinpath("trainval_ans2label.json")))
        self.label2ans = json.load(open(gqa_dir.joinpath("trainval_label2ans.json")))
        assert len(self.ans2label) == len(self.label2ans)
        for ans, label in self.ans2label.items():
            assert self.label2ans[label] == ans

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


class GQAEvaluator:
    def __init__(self, dataset: GQADataset):
        self.dataset = dataset
        self.punct        = [';', r"/", '[', ']', '"', '{', '}',
							 '(', ')', '=', '+', '\\', '_', '-',
							 '>', '<', '@', '`', ',', '?', '!']
        self.periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip   = re.compile("(\d)(\,)(\d)")
        self.articles     = ['a', 'an', 'the']

    def processArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            if word not in self.articles:
                outText.append(word)
        outText = " ".join(outText)
        return outText

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.periodStrip.sub("",
                                        outText,
                                        re.UNICODE)
        return outText

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            ans = self.processPunctuation(ans)
            ans = self.processArticle(ans)
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump the result to a GQA-challenge submittable json file.
        GQA json file submission requirement:
            results = [result]
            result = {
                "questionId": str,      # Note: it's a actually an int number but the server requires an str.
                "prediction": str
            }
        :param quesid2ans: A dict mapping question id to its predicted answer.
        :param path: The file path to save the json file.
        :return:
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                datum = self.dataset.id2datum[ques_id]
                label = datum['label']
                result.append({
                    'questionId': ques_id,
                    'prediction': ans,
                    'label' : label
                })
            json.dump(result, f, indent=4, sort_keys=True)
