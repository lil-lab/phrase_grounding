# Copyright: https://github.com/lil-lab/touchdown
from genericpath import exists
import pickle
from typing import List
import IPython
import numpy as np
import spacy

from nltk.corpus import stopwords

import time
import json
import os
from collections import defaultdict

from tqdm import tqdm

import dataset.dataset as dc
dataset_classes = dc.__dict__
import torch
from torchvision import transforms as pth_transforms
transform = pth_transforms.Compose([
    pth_transforms.ToTensor(),
    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

gpu = torch.device('cuda')
cpu = torch.device('cpu')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Loader:
    def __init__(self, data_dir, image_dir, target_dir, visualize: bool = False):
        self.data_dir = data_dir
        self.vocab = Vocabulary()
        self.max_length = 0
        self.datasets = {}
        self.image_dir = image_dir
        self.target_dir = target_dir
        self._visualize = visualize

    def load_json(self, filename, prefixes = ['main', 'pre', 'post']):
        path = os.path.join(self.data_dir, filename)
        data = []
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                for prefix in prefixes:
                    pano_type = prefix + '_pano'
                    heading_type = prefix + '_heading'
                    pitch_type = prefix + '_pitch'
                    center_type = prefix + '_static_center'
                    center = json.loads(obj[center_type])
                    
                    if prefix == "main":
                        prev_panoid = obj["route_panoids"][-2]
                        assert(obj["route_panoids"][-1] == obj[pano_type])
                    elif prefix == "pre":
                        prev_panoid = None
                    elif prefix == "post":
                        prev_panoid = None
                    if center == {'x': -1,'y': -1}:
                        continue

                    data.append({
                        'route_id': obj['route_id'],
                        'panoid': obj[pano_type], 
                        'prev_panoid': prev_panoid, 
                        'route_panoids': obj["route_panoids"],
                        'heading': obj[heading_type],
                        'pitch': obj[pitch_type],
                        'prefix': prefix,
                        'center': center,
                        'text': obj['td_location_text'],
                        'nav_text': obj['navigation_text'],
                        'full_text': obj['full_text'],
                    })
        return data

    def load_image_paths(self, data, image_mode: str):
        # todo: dirty implementation fix
        image_paths = []
        for data_obj in data:
            panoid = data_obj['panoid']
            route_id = data_obj['route_id']

            if image_mode == "unique_to_images":
                path = '{}{}.jpg'.format(self.image_dir, panoid)
                if not os.path.exists(path):
                    path = '{}{}.npy'.format(self.image_dir, panoid)
            elif image_mode == "unique_to_routes":
                path = '{}{}_{}.jpg'.format(self.image_dir, route_id, panoid)
                if not os.path.exists(path):
                    path = '{}{}_{}.npy'.format(self.image_dir, route_id, panoid)
            elif image_mode == "unique_to_routes_symlink":
                path = '{}{}_{}.jpg'.format(self.image_dir, route_id, panoid)
                path = os.readlink(path)
                if not os.path.exists(path):
                    path = '{}{}_{}.npy'.format(self.image_dir, route_id, panoid)
                    path = os.readlink(path)
                new_path = path.replace("../images", "/scratch/nk654/p-interactive-touchdown/data/sdr")
                if os.path.exists(new_path):
                    path = new_path
            else:
                raise ValueError("image_mode: {} is not supported.".format(image_mode))
            image_paths.append(path)
        return image_paths

    def load_target_paths(self, data):
        # todo: dirty implementation fix
        all_targets= ['{}{}.{}.npy'.format(self.target_dir, data_obj['route_id'], data_obj['panoid']) for data_obj in data]
        return [os.readlink(t)[1:] if not os.path.exists(t) else t for t in all_targets]
    
    def load_texts(self, data):
        return [data_obj['text'] for data_obj in data]

    def load_centers(self, data):
        return [data_obj['center'] for data_obj in data]

    def load_route_ids(self, data):
        return [data_obj['route_id'] for data_obj in data]
    
    def label_data(self, data):
        labels = []
        for data_obj in data:
            if data_obj["prefix"] == "main":
                label = "main"
            else:
                if data_obj["panoid"] == data_obj["route_panoids"][-2]:
                    label = "aligned"
                else:
                    label = "unaligned"
            labels.append(label)

        return labels

    def load_bbox(self, data, bbox_pkl_pathes):
        """
        Load bbox annotations
        """
        bboxes = []
        if not hasattr(self, '_bboxes_annotations'):
            self._bboxes_annotations = {}
            for p in bbox_pkl_pathes:
                print("Loading bboxes from {}".format(p))
                self._bboxes_annotations.update(pickle.load(open(p, "rb")))
        
        for data_obj in data:
            panoid = data_obj['panoid']
            route_id = data_obj['route_id']
            key = "{}_{}".format(route_id, panoid)
            if key in self._bboxes_annotations:
                bboxes.append(self._bboxes_annotations[key])
            else:
                #! can be used to detect missing annotation
                bboxes.append(None)

        return bboxes


    def build_vocab(self, texts, mode, get_obj_ref: bool = False, remove_stop_words: bool = False, **kwargs):
        '''Add words to the vocabulary'''
        
        parser = spacy.load('en_core_web_sm')
        ids = []
        seq_lengths = []
        all_refs = set()
        all_refs_ct = 0

        if get_obj_ref:
            obj_refs = []
            ref_lengths = []
            self._ignore_lists = [
                "touchdown",
            ]
            if remove_stop_words:
                self._ignore_lists += ["bear"] # todo: inconsistency
                self._ignore_lists += stopwords.words('english')

        for text in texts:
            line_ids = []
            doc = parser(text)
            self.max_length = max(self.max_length, len(doc))
            for word in doc:
                word = str(word).lower()
                word = self.vocab.add_word(word, mode)
                line_ids.append(self.vocab.word2idx[word])
            ids.append(line_ids)
            seq_lengths.append(len(doc))

            # get object references
            # todo: refactor as a separate function
            if get_obj_ref:
                refs = []
                for n in doc.noun_chunks:
                    all_refs.add(str(n))
                    if str(n).lower() not in self._ignore_lists:
                        refs.append([n.start, n.end])
                        all_refs_ct += 1
                refs = np.array(refs)
                obj_refs.append(refs)
                ref_lengths.append(refs.shape[0])
        
        text_ids = np.array([row + [0] * (self.max_length - len(row)) for row in ids])
        if not get_obj_ref:
            obj_refs, ref_lengths = None, None
        else:
            self.max_ref_length = np.max(ref_lengths)
            obj_refs_npy = np.zeros((len(obj_refs), self.max_ref_length, 2))
            for i, refs in enumerate(obj_refs):
                if len(refs) == 0:
                    print("No object mention found.") # todo
                    continue
                obj_refs_npy[i, :refs.shape[0], :] = refs

            obj_refs = obj_refs_npy

        return text_ids, seq_lengths, obj_refs, ref_lengths

    def build_vocab_bert(self, texts, clean_entities: bool = False, **kwargs):
        if clean_entities:
            # todo: update this argument during experiments
            self._ignore_lists = [
                "touchdown",
                "bear", 
            ]
            self._ignore_lists += stopwords.words('english')
        else:
            self._ignore_lists = []
        """    
        tokenized = self._tokenizer(texts, padding=True)
        text_ids = tokenized["input_ids"]
        attention_masks = tokenized["attention_mask"]
        """

        # https://stackoverflow.com/questions/66666525/how-to-map-token-indices-from-the-squad-data-to-tokens-from-bert-tokenizer
        # https://stackoverflow.com/questions/63932600/adding-entites-to-spacy-doc-object-using-berts-offsets
        # https://twitter.com/_inesmontani/status/1151447435195113472/photo/1
        from spacy.training import Alignment

        # extracting noun chunck
        parser = spacy.load('en_core_web_sm')
        ids, seq_lengths = [], []
        obj_refs, ref_lengths = [], []

        for i, text in enumerate(texts):
            # encode BERT objects
            gt_text_ids = self._tokenizer.encode(text)
            marked_text = self._tokenizer.bos_token + text + self._tokenizer.eos_token
            bert_tokens = self._tokenizer.tokenize(marked_text)
            text_ids = self._tokenizer.convert_tokens_to_ids(bert_tokens)
            assert(gt_text_ids == text_ids)
            ids.append(text_ids)
            seq_lengths.append(len(text_ids))
            self.max_length = max(self.max_length, len(text_ids))

            # resolve object references
            bert_refs = []
            doc = parser(text)
            spacy_tokens = [str(word) for word in doc]

            bert_tokens = [tk.replace("##", "").replace("Ġ", "") for tk in bert_tokens]
            align = Alignment.from_strings(bert_tokens[1:-1], spacy_tokens)
            
            # ! /home/nk654/local/anaconda3/envs/touchdown/lib/python3.7/site-packages/spacy/training
            # todo (broken case): text = "that doesn't have a multi story building on top of it. Touchdown is hiding at the middle bottom of those two red doors." 
            spacy_to_bert_offsets = np.array([0] + list(np.cumsum(align.y2x.lengths))[:-1]) + 1
            spacy_to_bert_ends = np.cumsum(align.y2x.lengths) + 1

            for n in doc.noun_chunks:
                if str(n).lower() not in self._ignore_lists:
                    # ! catch common falilure case of noun chunckers
                    # ? case 1: ~ hydrant
                    if str(n).endswith("fire"):
                        if n.end < len(spacy_tokens) and spacy_tokens[n.end] == 'hydrant':
                            n.end += 1
                    # ? case 2: ~ door handle
                    if str(n).endswith("door"):
                        if n.end < len(spacy_tokens) and spacy_tokens[n.end] == 'handle':
                            n.end += 1
                    # ? case 3: ~ metal pole
                    if str(n).endswith("metal"):
                        if n.end < len(spacy_tokens) and spacy_tokens[n.end] == 'pole':
                            n.end += 1

                    bert_ref = [spacy_to_bert_offsets[n.start], spacy_to_bert_ends[n.end-1]]
                    bert_refs.append(bert_ref)            

            bert_refs = np.array(bert_refs)
            obj_refs.append(bert_refs)
            ref_lengths.append(bert_refs.shape[0])

        text_ids = np.array([row + [self._tokenizer.pad_token_id] * (self.max_length - len(row)) for row in ids])
        self.max_ref_length = np.max(ref_lengths)
        obj_refs_npy = np.zeros((len(obj_refs), self.max_ref_length, 2))
        for i, refs in enumerate(obj_refs):
            if len(refs) == 0:
                print("No object mention found.") # todo
                continue
            obj_refs_npy[i, :refs.shape[0], :] = refs
        
        return text_ids, seq_lengths, obj_refs_npy, ref_lengths

    def build_dataset(self, file, gaussian_target, sample_used, dataset_class, use_bert: bool = False, image_mode: str = "unique_to_images", remove_invalid_targets: bool = False, use_bbox: bool = False, bbox_pkl_pathes: List = [], **kwargs):
        mode, ext = file.split('.')
        print('[{}]: Start loading JSON file...'.format(mode))
        data = self.load_json(file)
        data_size = len(data)

        if isinstance(sample_used, tuple):
            start, end = sample_used
            data = data[start:end]
            num_samples = end - start
        elif isinstance(sample_used, int) or isinstance(sample_used, float):
            if sample_used <= 1:
                num_samples = int(len(data) * sample_used)
            elif 1 < sample_used < len(data):
                num_samples = int(sample_used)
            else:
                raise ValueError
            data = data[:num_samples]
        else:
            raise ValueError
        print('[{}]: Using {} ({}%) samples'.format(mode, num_samples, num_samples / data_size * 100))

        data = self.validate_data(data, image_mode, remove_invalid_targets=remove_invalid_targets)
        image_paths = self.load_image_paths(data, image_mode)
        centers = self.load_centers(data)
        target_paths = self.load_target_paths(data)
        route_ids = self.load_route_ids(data)
        # label data according to the transformation status 
        data_labels = self.label_data(data)

        if use_bbox:
            bboxes = self.load_bbox(data, bbox_pkl_pathes)
        else:
            bboxes = None

        # todo: replace this with BERT tokenizer later :)
        print('[{}]: Building vocab from text data...'.format(mode))
        texts = self.load_texts(data)
        self._raw_text = texts

        # todo: not tested :/
        if use_bert: 
            if not hasattr(self, '_tokenizer'):
                """
                https://huggingface.co/transformers/glossary.html
                todo: check if we can ignore token_type_ids
                https://huggingface.co/transformers/preprocessing.html 
                """
                if "tokenizer_path" not in kwargs:
                    #self._tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')    # Download vocabulary from S3 and cache.
                    from transformers import BertTokenizerFast
                    self._tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
                    self._tokenizer.bos_token = "[CLS] "
                    self._tokenizer.eos_token = " [SEP]"
                else:
                    from transformers import BertTokenizerFast, RobertaTokenizerFast
                    tokenizer_classes = {
                        "bert_model": BertTokenizerFast,
                        "roberta_model": RobertaTokenizerFast
                    }
                    self._tokenizer = tokenizer_classes[kwargs["tokenizer_class"]].from_pretrained(kwargs["tokenizer_path"])
                    if kwargs["tokenizer_class"] == "bert_model":
                        self._tokenizer.bos_token = "[CLS] "
                        self._tokenizer.eos_token = " [SEP]"
            texts, seq_lengths, obj_refs, ref_lengths = self.build_vocab_bert(texts)
        else:
            texts, seq_lengths, obj_refs, ref_lengths = self.build_vocab(texts, mode, **kwargs)

        # todo load object bounding boxes
        print('[{}]: Building dataset...'.format(mode))
        kwargs.update({
            "image_paths": image_paths,
            "texts": texts,
            "seq_lengths": seq_lengths,
            "target_paths": target_paths,
            "centers": centers,
            "gaussian_target": gaussian_target,
            "route_ids": route_ids,
            "data_labels": data_labels,
            "obj_refs": obj_refs,
            "ref_lengths": ref_lengths,
            "raw_text": self._raw_text,
            "bboxes": bboxes,
            "visualize": self._visualize,
        })
        dataset = dataset_classes[dataset_class](**kwargs)
        self.datasets[mode] = dataset
        print('[{}]: Finish building dataset...'.format(mode))
    
    def validate_data(self, data, image_mode: str, remove_invalid_targets: bool = False):
        if remove_invalid_targets:
            infile = open("../data/invalid_targets.txt", "r")
            invalid_image_lists = [l.strip() for l in infile.readlines()]

        new_data = []
        for data_obj in data:
            panoid = data_obj['panoid']
            route_id = data_obj['route_id']

            if image_mode == "unique_to_images":
                jpg_image_path = '{}{}.jpg'.format(self.image_dir, panoid)
                npy_image_path = '{}{}.npy'.format(self.image_dir, panoid)
            elif image_mode == "unique_to_routes" or image_mode == "unique_to_routes_symlink":
                jpg_image_path = '{}{}_{}.jpg'.format(self.image_dir, route_id,panoid)
                npy_image_path = '{}{}_{}.npy'.format(self.image_dir, route_id, panoid)
            else:
                raise ValueError("image_mode: {} is not supported.".format(image_mode))

            if remove_invalid_targets:
                target_path = '{}{}.{}.npy'.format(self.target_dir, route_id, panoid)
                isvalid_target = False if target_path in invalid_image_lists else True
                if not isvalid_target:
                    print("{} not valid target".format(target_path))
            else:
                isvalid_target = True

            if os.path.exists(jpg_image_path) or os.path.exists(npy_image_path):
                if isvalid_target:
                    new_data.append(data_obj)
            else:
                # symlink
                try:
                    jpg_image_path = os.readlink(jpg_image_path)
                except:
                    pass
                try:
                    npy_image_path = os.readlink(jpg_image_path)
                except:
                    pass
                if os.path.exists(jpg_image_path) or os.path.exists(npy_image_path):
                    if isvalid_target:
                        new_data.append(data_obj)
                else:
                    print("{} missing".format(panoid))
       
        return new_data

    def validate_index(self, data, is_transformer: bool):
        invalid_index = []
        for i, data_obj in enumerate(data):
            panoid = data_obj['panoid']

            if is_transformer:
                image_path = '{}{}.jpg'.format(self.image_dir, panoid)
            else:
                image_path = '{}{}.npy'.format(self.image_dir, panoid)
            if not os.path.exists(image_path):
                invalid_index.append(i)
                print("{} missing".format(image_path))
        return invalid_index

class Vocabulary:
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}

    def add_word(self, word, mode):
        if word not in self.word2idx and mode in ('train', 'dev', 'debug', 'debug_bigger', "debug_dev"):
            idx = len(self.idx2word)
            self.idx2word[idx] = word
            self.word2idx[word] = idx
            return word
        elif word not in self.word2idx and mode == 'test':
            return '<unk>'
        else:
            return word

    def __len__(self):
        return len(self.idx2word)



