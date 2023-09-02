from copyreg import pickle
from tkinter import E
from .base_dataset import BaseDataset
import os, pickle
import io
import ast
import cv2
import copy
import re
import random
import time
import pandas as pd
import pyarrow as pa
import numpy as np
from tqdm import tqdm
from PIL import Image
from ast import literal_eval
import random
import pyarrow as pa
import pyarrow.compute as pc
from IPython import embed
from typing import List, Tuple, Dict

import torch
from torch import nn
from torch.utils.data.sampler import Sampler

from pycocotools import mask as coco_mask


pooler = nn.functional.adaptive_avg_pool2d


def box_rel_to_abs(bboxes, img_size):
    img_h, img_w = img_size
    bboxes[:, [0,2]] *= img_w
    bboxes[:, [1,3]] *= img_h
    bboxes[:, 2] += bboxes[:, 0]
    bboxes[:, 3] += bboxes[:, 1]
    return bboxes

def box_abs_to_rel(bboxes, img_size):
    img_h, img_w = img_size
    bboxes[:, [0,2]] /= img_w
    bboxes[:, [1,3]] /= img_h
    bboxes[:, 2] -= bboxes[:, 0]
    bboxes[:, 3] -= bboxes[:, 1]
    return bboxes

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x
    b = np.array([(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)])
    return b


def polygons_rel_to_abs(polygons: List, img_size: Tuple):
    h, w = img_size
    for i in range(len(polygons)):
        for j in range(len(polygons[i])):
            if j % 2 == 0:
                polygons[i][j] = polygons[i][j]  * float(w)
            else:
                polygons[i][j]  = polygons[i][j]  * float(h)
    return polygons


def create_target_tensor(annotations: List, img_size: Tuple, orig_img_size: Tuple = None, is_polygon: bool = False):

    if orig_img_size is not None:
        orig_img_size, target_img_size  = orig_img_size, img_size
    else:
        orig_img_size, target_img_size  = img_size, img_size

    if is_polygon:
        annotations = copy.deepcopy(annotations)
        annotations = polygons_rel_to_abs(annotations, target_img_size)
        rles = coco_mask.frPyObjects(annotations, target_img_size[0], target_img_size[1])
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        target = mask.any(dim=2)            
    else:
        target = torch.zeros(orig_img_size, dtype=torch.float) 
        for b in annotations:
            b = np.array(b)
            b[[0,2]] *= orig_img_size[1]
            b[[1,3]] *= orig_img_size[0]
            x, y, w, h = b.astype(int)
            target[y:y+h, x:x+w] = 1.
    
    # resize target image using interpolation
    if orig_img_size != target_img_size:
        target = target.unsqueeze(0).unsqueeze(0)
        target = pooler(target, output_size=target_img_size).squeeze(0).squeeze(0)

    # normalize the target image so that the sum of the values to be 1.
    target = target / torch.sum(target)

    return target

# this function is directly borrowed from 
# https://github.com/ashkamath/mdetr/blob/main/datasets/coco.py#L64
def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def create_gaussian_target(center, sigma=3.0, img_size: Tuple = (100, 464)):
    img_h, img_w = img_size
    #
    if img_size == (100, 464) and (sigma == 3.0):
        slack = 15
        x_min = np.max([np.floor(center['x'] * 464 - slack).astype(int),  0])
        x_max = np.min([np.ceil(center['x'] * 464 + slack).astype(int),  464])
        y_min = np.max([np.floor(center['y'] * 100 - slack).astype(int),  0])
        y_max = np.min([np.ceil(center['y'] * 100 + slack).astype(int),  100])

        target = np.zeros((100, 464))
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                pixel_val = (1 / (2 * np.pi * sigma ** 2)) * \
                np.exp(-((x - center['x'] * img_w) ** 2 + (y - center['y'] * img_h) ** 2) / (2 * sigma ** 2))
                pixel_val = pixel_val if pixel_val > 1e-5 else 0.
                target[y, x] = pixel_val        
    else:
        def gaussian(x, y, center, sigma):
            pixel_val = (1 / (2 * np.pi * sigma ** 2)) * \
                np.exp(-((x - center['x'] * img_w) ** 2 + (y - center['y'] * img_h) ** 2) / (2 * sigma ** 2))
            return pixel_val if pixel_val > 1e-5 else 0.0

        target = [[gaussian(x, y, center, sigma) for x in range(img_w)] for y in range(img_h)]
    target = torch.as_tensor(target)
    target /= torch.sum(target)

    return target

def create_positive_map(tokenized, tokens_positive, max_text_len=80):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), max_text_len), dtype=torch.float)
    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos : end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)

class VLGDataset(BaseDataset):
    def __init__(self, *args, split="", dstype="", _configs: Dict={}, **kwargs):
        assert split in ["train", "val", "test"]
        data_dir = args[0]
        self.transform_keys = args[1]
        
        if split == "train":
            names = [f.replace(".arrow", "") for f in os.listdir(data_dir) if f.endswith(".arrow") and "train" in f] 
        elif split == "val":
            names = []
            if dstype == "touchdown_sdr":
                names += [f.replace(".arrow", "") for f in os.listdir(data_dir) if f.endswith(".arrow") and "tune" in f] 
                names += [f.replace(".arrow", "") for f in os.listdir(data_dir) if f.endswith(".arrow") and "dev" in f] 
            elif dstype == "f30k_refgame":
                names += [f.replace(".arrow", "") for f in os.listdir(data_dir) if f.endswith(".arrow") and "dev" in f] 
            elif dstype == "tangram_refgame":
                names += [f.replace(".arrow", "") for f in os.listdir(data_dir) if f.endswith(".arrow") and "val" in f] 
        elif split == "test":
            names = []
            if dstype == "touchdown_sdr":
                names += [f.replace(".arrow", "") for f in os.listdir(data_dir) if f.endswith(".arrow") and "tune" in f] 
                #names += [f.replace(".arrow", "") for f in os.listdir(data_dir) if f.endswith(".arrow") and "dev" in f] 
                #names += [f.replace(".arrow", "") for f in os.listdir(data_dir) if f.endswith(".arrow") and "test" in f] 
            elif dstype == "f30k_refgame":
                names += [f.replace(".arrow", "") for f in os.listdir(data_dir) if f.endswith(".arrow") and "dev" in f] 
                names += [f.replace(".arrow", "") for f in os.listdir(data_dir) if f.endswith(".arrow") and "test" in f] 
            elif dstype == "tangram_refgame":
                names += [f.replace(".arrow", "") for f in os.listdir(data_dir) if f.endswith(".arrow") and "val" in f] 
                names += [f.replace(".arrow", "") for f in os.listdir(data_dir) if f.endswith(".arrow") and "dev" in f] 
                names += [f.replace(".arrow", "") for f in os.listdir(data_dir) if f.endswith(".arrow") and "test" in f] 

        super().__init__(*args, **kwargs, names=names, is_touchdown_sdr= (dstype=="touchdown_sdr"))
        assert(len(self.transform_keys) == 1)

        # remove bounding boxes for ablation
        if split == "train": 
            assert not ((_configs["training_phrase_grounding_ratio"] != 1.0) and  (_configs["training_grounding_annotated_image_ratio"] != 1.0)), "reduce annotated phrases with grounding annotations or annnotated images but not both." 
        
            if _configs["training_phrase_grounding_ratio"] != 1.0:
                self.reduce_bounding_boxes(training_phrase_grounding_ratio=_configs["training_phrase_grounding_ratio"])
            elif _configs["training_grounding_annotated_image_ratio"] != 1.0:
                self.reduce_annotated_images(annotation_ratio=_configs["training_grounding_annotated_image_ratio"], is_flickr30k=(dstype =="f30k_refgame"))

        # If the data split is 'train' and the model class is 'vilt_aligner_probe', 
        # we remove examples without phrase grounding annotations. This is particularly useful for probing experiments.
        if split == "train" and _configs["model_cls"] == "vilt_aligner_probe":
            self.drop_examples_without_grounding_annotations()

        return names
  
    def get_raw_image(self, index, image_key="image"):
        # if no image is pre-compured on the table directly read from the image path?
        image_entry = self.table[image_key][index].as_py()
        image_path = self.table["image_path"][index].as_py()
        alternate_image_path = image_path.replace("/scratch/n654/p-interactive-touchdown", "/home/nk654/projects/p-interactive-touchdown")
        if image_entry is None:
            try:
                if not os.path.exists(image_path):
                    image_path = alternate_image_path
                    assert(os.path.exists(image_path))
            except:
                raise ValueError(f"{image_path} does not exisit. It is likely you are runnning experiments on the wrong node.")
            with open(image_path, "rb") as fp:
                image_entry = fp.read()
        
        image_bytes = io.BytesIO(image_entry)
        image_bytes.seek(0)
        image = Image.open(image_bytes).convert("RGB")

        return {
            "image": image,
            "image_size": image.size[::-1],
            "image_path": image_path
        } 

    def apply_image_transformations(self, image_info, bbox_info):
        transformed_bboxes, transformed_polygons = [], []

        if "mdetr" in self.transform_keys[0]: 
            image = image_info["image"]

            for v in bbox_info.values():
                transformed_bboxes += v["bboxes"]
            
            # converting bboxes to absolute coordinates
            torch_bboxes = torch.as_tensor(transformed_bboxes)
            img_w, img_h = image.size
            img_size = (img_h, img_w)
            if len(torch_bboxes) != 0:
                torch_bboxes = box_rel_to_abs(torch_bboxes, img_size)
                targets = {"boxes": torch_bboxes}
            else:
                targets = {}

            # apply transform
            # bboxes are  converted to cxcywh format
            image_tensor, targets = self.transforms[0](image, targets)

            # converting bboxes / sdr center back to relative coordinates 
            img_h, img_w = image_tensor.size()[1:]
            img_size = (img_h, img_w)
            if "boxes" in targets:
                transformed_bboxes = targets["boxes"]
            else:
                transformed_bboxes = torch_bboxes
        else:
            image_tensor = self.transforms[0](image_info["image"])
            for v in bbox_info.values():
                transformed_bboxes += v["bboxes"]
                if "segmentation" in v:
                    transformed_polygons += v["segmentation"]
            transformed_bboxes = torch.as_tensor(transformed_bboxes)
            transformed_polygons = transformed_polygons
        
        transformed = {
            "boxes": transformed_bboxes,
            "polygons": transformed_polygons
        }

        return image_tensor, transformed

    def process_bbox_info(self, image_tensor, bbox_info, transformed_bboxes, tokenized, transformed_polygons = None, img_size = None, generate_target_tensors: bool = True, verbose: bool = False, debug: bool = False, debug_args: Dict = None):
        # compile target information
        bbox_tensors, bbox_coords, positive_maps, tokens_positives, phrase_ids = None, None, None, None, None 

        if len(bbox_info) != 0:
            target_size = image_tensor.size()[1:] if img_size is None else img_size
            bbox_tensors, bbox_coords, polygon_coords = [], [], []
            tokens_positives, phrase_ids = [], []
            bbox_offset, polygon_offset = 0, 0



            for k, v in bbox_info.items():
                # reformat bbox information in the data table
                num_bboxes = len(v["bboxes"])
                bboxes = transformed_bboxes[bbox_offset: bbox_offset+num_bboxes]
                bbox_offset += num_bboxes
                bbox_coords.append(bboxes.numpy())

                if transformed_polygons is not None:
                    num_polygons = len(v["segmentation"])
                    polygons = transformed_polygons[polygon_offset: polygon_offset+num_polygons]
                    polygon_offset += num_polygons
                    polygon_coords.append(polygons)

                tokens_positive = v["tokens_positive"]
                phrase_ids.append(k)
                tokens_positives.append(tokens_positive)
                if generate_target_tensors:
                    if self.is_touchdown_sdr:
                        img_size, orig_img_size = (100, 464), (800, 3712)
                    else:
                        img_size, orig_img_size = target_size, None

                    if transformed_polygons is None:
                        target_tensor = create_target_tensor(bboxes, img_size=target_size)
                    else:
                        target_tensor = create_target_tensor(polygons, img_size=target_size, is_polygon = True)

                    bbox_tensors.append(target_tensor.unsqueeze(0))

            positive_maps = create_positive_map(tokenized, tokens_positives, self.max_text_len).bool()  
            if generate_target_tensors:
                bbox_tensors = torch.cat(bbox_tensors, 0).float()

            if verbose:
                for i in range(positive_maps.shape[0]):
                    pos_tokens = self.tokenizer.decode(torch.tensor(tokenized["input_ids"])[positive_maps[i, :]])
                    print(pos_tokens)
        
            # visualization
            if debug:
                index = debug_args["index"]
                caption = debug_args["caption"] if "caption" in debug_args else self.get_text(index)["text"][0] 
                image_only = debug_args["image_only"] if "image_only" in debug_args else False

                folder_path = "/home/nk654/projects/p-interactive-touchdown/models/Touchdown-ViLT/images/debug/mdetr_image_augumentation"
                os.makedirs(folder_path, exist_ok=True)
                x = image_tensor.new(*image_tensor.size())
                # reference: https://discuss.pytorch.org/t/simple-way-to-inverse-normalize-a-batch-of-input-variable/12385
                x[0, :, :] = image_tensor[0, :, :] * 0.5 + 0.5
                x[1, :, :] = image_tensor[1, :, :] * 0.5 + 0.5
                x[2, :, :] = image_tensor[2, :, :] * 0.5 + 0.5
                image_numpy = (x.permute(1,2,0).numpy() * 255).astype(int)
                r,g,b = cv2.split(image_numpy)
                img_bgr = cv2.merge([b,g,r])
                width = img_bgr.shape[1] 
                height = img_bgr.shape[0] 

                # draw bounding boxes
                num_iters = 1 if image_only else len(bbox_coords)
                for i in range(num_iters):
                    if generate_target_tensors:
                        bbox_mask = bbox_tensors[i]
                        bbox_mask = bbox_mask / torch.max(bbox_mask)
                        bbox_mask = bbox_mask.float().numpy()
                        heatmap_img = cv2.cvtColor(bbox_mask, cv2.COLOR_BGR2RGB)
                        dim = (width, height)
                        heatmap_img  = cv2.resize(heatmap_img, dim, interpolation = cv2.INTER_LINEAR)
                        heatmap_img = (heatmap_img * 255).astype(int)
                        fin = cv2.addWeighted(heatmap_img, 0.3, img_bgr, 0.7, 0)
                    else:
                        fin = img_bgr
                    
                    if not image_only:
                        # apply segmentation masks
                        if "segmentation_masks" in debug_args:
                            mask = debug_args["segmentation_masks"].numpy() 
                            mask /= np.max(mask)  
                            mask *= 255
                            mask = cv2.cvtColor(mask[0, ...], cv2.COLOR_GRAY2BGR)
                            mask = cv2.convertScaleAbs(mask)
                            mask_img = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                            mask_img = mask_img.astype(np.int32)
                            fin = cv2.addWeighted(mask_img, 0.7, fin, 0.3, 0)

                        for j, bbox in enumerate(bbox_coords[i]):
                            if "mdetr" in self.transform_keys[0]:
                                bbox = box_cxcywh_to_xyxy(bbox)
                                bbox[[0,2]] *= width
                                bbox[[1,3]] *= height
                                start_point = np.array([bbox[0], bbox[1]])
                                end_point = np.array([bbox[2], bbox[3]])
                            else:
                                bbox[[0,2]] *= width
                                bbox[[1,3]] *= height
                                start_point = np.array([bbox[0], bbox[1]])
                                end_point = np.array([bbox[0] + bbox[2], bbox[1] + bbox[3]])
                            start_point = (start_point).astype(int)
                            end_point = (end_point).astype(int)
                            fin = cv2.rectangle(fin, start_point, end_point, (255, 0, 0), 3) 

                    # save image
                    save_path = "{}/{}_{}.jpg".format(folder_path, str(self.table["identifier"][index]), i)
                    cv2.imwrite(save_path, fin)
                    del fin

                    # print out captions / phrases
                    start, end = tokens_positives[i][0]
                    msg = "{} : {}\n {} ({}, {})".format(caption[start:end], caption, save_path, width, height)
                    print(msg)
                    print("")

        return (bbox_tensors, bbox_coords, positive_maps, tokens_positives, phrase_ids) if transformed_polygons is None else (bbox_tensors, bbox_coords, polygon_coords, positive_maps, tokens_positives, phrase_ids)
 
    def get_image_and_target(self, index, tokenized, image_key="image", 
    verbose: bool = False):
        image_info = self.get_raw_image(index, image_key=image_key)
        bbox_info = literal_eval(str(self.table["bbox_info"][index]))
        
        # apply image augumentation
        image_tensor, transformed = self.apply_image_transformations(image_info, bbox_info)
        transformed_bboxes = transformed["boxes"]

        # format bounding box annotations
        bbox_tensors, bbox_coords, positive_maps, tokens_positives, phrase_ids = self.process_bbox_info(image_tensor, bbox_info, transformed_bboxes, tokenized, debug = False, debug_args = {"index": index})

        return {
            "image": [image_tensor],
            "image_path": image_info["image_path"],
            "image_orig_size": image_info["image_size"],
            "bbox_tensors": bbox_tensors, 
            "bbox_coords": bbox_coords,
            "bbox_positive_maps": positive_maps,
            "bbox_token_postives": tokens_positives,
            "phrase_ids": phrase_ids,
        }
 
    def get_text(self, index, verbose=False):
        # get caption
        text = str(self.table["caption"][index])
        tokenized = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )

        return {
            "text": (text, tokenized),
        }

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        identifier = str(self.table["identifier"][index])
        split = str(self.table["split"][index])        
        text = self.get_text(index)["text"]
        tokenized = text[1]
        image_target_info = self.get_image_and_target(index, text[1])

        return {
            "text": text,
            "identifier": identifier,
            "split": split,
            "image": image_target_info["image"],
            "picture_path": image_target_info["image_path"], 
            "picture_orig_size": image_target_info["image_orig_size"], 
            "bbox_tensors": image_target_info["bbox_tensors"], 
            "bbox_coords": image_target_info["bbox_coords"],
            "bbox_positive_maps": image_target_info["bbox_positive_maps"], 
            "bbox_token_postives": image_target_info["bbox_token_postives"],
            "phrase_ids": image_target_info["phrase_ids"],
        }

    def debug(self):
        from transformers import (
            DataCollatorForLanguageModeling,
            DataCollatorForWholeWordMask,
            BertTokenizer,
            BertTokenizerFast,
        )
        self.tokenizer = BertTokenizerFast.from_pretrained(
            "bert-base-uncased", #do_lower_case="uncased" 
        )
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False 
        )
        batch_1 = self.__getitem__(0)
        batch_2 = self.__getitem__(1)
        collated_batch = self.collate([batch_1, batch_2], collator)

        for i in range(len(self.table)):
           batch = self.__getitem__(i)
           #print(batch["picture_path"], batch["label_t"], batch["text"][0])

    def reduce_bounding_boxes(self, training_phrase_grounding_ratio: float):
        """
        Drop phrases (not bounding boxes) based on training_phrase_grounding_ratio.
        """
        num_remained_phrases = 0
        if training_phrase_grounding_ratio > 0.5:
            phrase_counter = 0
            for column_name in self.table.column_names:
                column_data = self.table[column_name]
                if column_name == "bbox_info":
                    bbox_info_arr = self.table["bbox_info"].to_numpy()
                    for i in range(len(bbox_info_arr)):
                        bbox_info = ast.literal_eval(bbox_info_arr[i])
                        for k in bbox_info.keys():
                            phrase_counter += 1
            keep_inds = set(random.sample([i for i in range(phrase_counter)], int(phrase_counter * training_phrase_grounding_ratio)))
        else:
            interval = int(1 / training_phrase_grounding_ratio)

        from tqdm import tqdm
        columns = []
        phrase_counter = 0
        num_remained_phrases = 0
        for column_name in self.table.column_names:
            column_data = self.table[column_name]
            if column_name == "bbox_info":
                bbox_info_arr = self.table["bbox_info"].to_numpy()
                new_bbox_info_arr = []
                for i in tqdm(range(len(bbox_info_arr))):
                    bbox_info = ast.literal_eval(bbox_info_arr[i])
                    new_bbox_info = {}
                    for k in bbox_info.keys():
                        phrase_counter += 1
                        discard_phrase = True
                        if training_phrase_grounding_ratio > 0.5:
                            if phrase_counter in keep_inds:
                                discard_phrase = False
                        else:
                            if phrase_counter % interval == 0:
                                discard_phrase = False
                        if not discard_phrase:
                            new_bbox_info[k] = bbox_info[k]
                            num_remained_phrases += 1
                    new_bbox_info_arr.append(str(new_bbox_info))
                column_data = pa.array(new_bbox_info_arr)
            columns.append(column_data)
        
        print("keeping {} / {} phrases".format(num_remained_phrases, phrase_counter))
        self.table = pa.Table.from_arrays(
            columns, 
            schema=self.table.schema
        )

    def reduce_annotated_images(self, annotation_ratio: float, is_flickr30k: bool=False):
        if is_flickr30k:
            # flickr30k
            all_img_list = list(self.table["image_path"].to_numpy())
            all_img_set = set(all_img_list)
            num_images = len(all_img_set)
            num_keep_imgs = int(num_images * annotation_ratio)
            sampled_images = random.sample(all_img_set, num_keep_imgs)
            keep_inds = [i for i in range(len(all_img_list)) if all_img_list[i] in sampled_images]
        else:
            # other dataset
            num_images = len(self.table["image"])
            keep_inds = set(random.sample([i for i in range(num_images)], int(num_images * annotation_ratio)))

        columns = []
        for column_name in self.table.column_names:
            column_data = self.table[column_name]
            if column_name == "bbox_info":
                bbox_info_arr = self.table["bbox_info"].to_numpy()
                reduced_bbox_info_arr = [
                    bbox_info_arr[i] if i in keep_inds else str({}) for i in range(len(bbox_info_arr)) 
                ]
                column_data = pa.array(reduced_bbox_info_arr)
                print("Keep phrase grounding annotations of {} / {} images ".format(len(keep_inds), num_images))
            columns.append(column_data)

        self.table = pa.Table.from_arrays(
            columns, 
            schema=self.table.schema
        )
    
    def drop_examples_without_grounding_annotations(self):
        """Remove rows from the PyArrow table which lack grounding annotations."""

        # Get the indices of rows with non-empty grounding annotations
        keep_inds = [i for i, bbox_info in enumerate(self.table["bbox_info"]) 
                    if len(literal_eval(str(bbox_info))) != 0]

        # Convert the table to a pandas DataFrame
        df = self.table.to_pandas()

        # Delete the original table to free up memory
        self.table = None

        # Keep only the rows with non-empty grounding annotations
        df = df.loc[keep_inds]

        # Convert the filtered DataFrame back to a PyArrow table
        filtered_table = pa.Table.from_pandas(df)

        # Update the table attribute with the filtered table
        self.table = filtered_table

class TouchdownSDRDataset(VLGDataset):
    def __init__(self, *args, split="", _configs: Dict={}, **kwargs):
        super().__init__(*args, split=split, dstype="touchdown_sdr", _configs=_configs, **kwargs)

    def apply_image_transformations(self, image_info, bbox_info, sdr_coords):
        transformed_bboxes = []

        if "mdetr" in self.transform_keys[0]: 
            image = image_info["image"]

            for v in bbox_info.values():
                transformed_bboxes += v["bboxes"]

            # appending fake box for sdr coords
            sdr_bbox = [sdr_coords["x"], sdr_coords["y"], 1e-4, 1e-4]
            transformed_bboxes.append(sdr_bbox)
            
            # converting bboxes to absolute coordinates
            torch_bboxes = torch.as_tensor(transformed_bboxes)
            img_w, img_h = image.size
            img_size = (img_h, img_w)
            torch_bboxes = box_rel_to_abs(torch_bboxes, img_size)
            targets = {"boxes": torch_bboxes}

            # apply transform
            # bboxes are  converted to cxcywh format
            image_tensor, targets = self.transforms[0](image, targets)

            # converting bboxes / sdr center back to relative coordinates 
            img_h, img_w = image_tensor.size()[1:]
            img_size = (img_h, img_w)
            transformed_bboxes = targets["boxes"]
            # transformed_bboxes = box_abs_to_rel(targets["boxes"], img_size)
            sdr_coords = {
                "x": transformed_bboxes[-1][0].item(),
                "y": transformed_bboxes[-1][1].item(),
            }
            transformed_bboxes = transformed_bboxes[:-1]
        else:
            image_tensor = self.transforms[0](image_info["image"])
            for v in bbox_info.values():
                transformed_bboxes += v["bboxes"]
            transformed_bboxes = torch.as_tensor(transformed_bboxes)

        return image_tensor, transformed_bboxes, sdr_coords
    
    
    def process_sdr_info(self, sdr_info, transformed_sdr_coords, tokenized, verbose: bool = False, debug: bool = False, debug_args: Dict = None):
        
        # create target tensor
        target_tensor = create_gaussian_target(transformed_sdr_coords)

        # create positive map for touchdown mentions
        tokens_positive = sdr_info["tokens_positive"]
        positive_map = create_positive_map(tokenized, [tokens_positive], self.max_text_len)
        positive_map = torch.sum(positive_map, 0).bool()
        positive_map_list = positive_map.tolist()
        tokenized["positive_map"] = positive_map_list

        if verbose:
            pos_tokens = self.tokenizer.decode(torch.tensor(tokenized["input_ids"])[positive_map])
            print(pos_tokens)
        
        if debug:
            index = debug_args["index"]
            image_tensor = debug_args["image_tensor"]
            caption = self.get_text(index)["text"][0]
            print(caption)
            folder_path = "/home/nk654/projects/p-interactive-touchdown/models/Touchdown-ViLT/images/debug/mdetr_image_augumentation"
            os.makedirs(folder_path, exist_ok=True)
            x = image_tensor.new(*image_tensor.size())
            # reference: https://discuss.pytorch.org/t/simple-way-to-inverse-normalize-a-batch-of-input-variable/12385
            x[0, :, :] = image_tensor[0, :, :] * 0.5 + 0.5
            x[1, :, :] = image_tensor[1, :, :] * 0.5 + 0.5
            x[2, :, :] = image_tensor[2, :, :] * 0.5 + 0.5
            image_numpy = (x.permute(1,2,0).numpy() * 255).astype(int)
            r,g,b = cv2.split(image_numpy)
            img_bgr = cv2.merge([b,g,r])

            # process sdr heatmap
            bbox_mask = target_tensor
            bbox_mask = bbox_mask.float().numpy()
            heatmap_img = cv2.cvtColor(bbox_mask, cv2.COLOR_GRAY2BGR)
            heatmap_img = heatmap_img * 255 / np.max(heatmap_img)
            heatmap_img = cv2.convertScaleAbs(heatmap_img)
            heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
            heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)

            width = img_bgr.shape[1] 
            height = img_bgr.shape[0] 
            dim = (width, height)
            heatmap_img  = cv2.resize(heatmap_img, dim, interpolation = cv2.INTER_LINEAR)
            heatmap_img = (heatmap_img * 255).astype(int)
            fin = cv2.addWeighted(heatmap_img, 0.3, img_bgr, 0.7, 0)

            # save image
            save_path = "{}/{}_sdr.jpg".format(folder_path, str(self.table["identifier"][index]))
            cv2.imwrite(save_path, fin)
            del fin
            print(save_path)

        return target_tensor, transformed_sdr_coords, tokenized

    
    def get_image_and_target(self, index, tokenized, image_key="image", 
    verbose: bool = False):
        image_info = self.get_raw_image(index, image_key=image_key)
        bbox_info = literal_eval(str(self.table["bbox_info"][index]))
        sdr_info = literal_eval(str(self.table["sdr_info"][index]))

        # apply image augumentation
        image_tensor, transformed_bboxes, transformed_sdr_coords = self.apply_image_transformations(image_info, bbox_info, sdr_info["coords"])

        # format bounding box annotations
        bbox_tensors, bbox_coords, positive_maps, tokens_positives, phrase_ids = self.process_bbox_info(image_tensor, bbox_info, transformed_bboxes, tokenized, debug=False, debug_args = {"index": index})

        # format sdr annotations
        target_tensor, target_coords, tokenized = self.process_sdr_info(sdr_info, transformed_sdr_coords, tokenized, debug=False, debug_args = {"index": index, "image_tensor": image_tensor})

        return {
            "image": [image_tensor],
            "image_path": image_info["image_path"],
            "image_orig_size": image_info["image_size"],
            "target_tensor": target_tensor, 
            "target_coords": torch.tensor([target_coords["x"], target_coords["y"]]),
            "bbox_tensors": bbox_tensors, 
            "bbox_coords": bbox_coords,
            "bbox_positive_maps": positive_maps,
            "bbox_token_postives": tokens_positives,
            "phrase_ids": phrase_ids,
        }

    def __getitem__(self, index):
        identifier = str(self.table["identifier"][index])
        split = str(self.table["split"][index])    
        caption_id = str(self.table["caption_id"][index])
        is_propagated = self.table["is_propagated"][index].as_py()     
        text = self.get_text(index)["text"]
        tokenized = text[1]
        image_target_info = self.get_image_and_target(index, tokenized)

        return {
            "text": text,
            "identifier": identifier,
            "split": split,
            "caption_id": caption_id, 
            "is_propagated": is_propagated,
            "image": image_target_info["image"],
            "picture_path": [image_target_info["image_path"]], 
            "picture_orig_size": image_target_info["image_orig_size"], 
            "target": image_target_info["target_tensor"],
            "target_coords": image_target_info["target_coords"], 
            "bbox_tensors": image_target_info["bbox_tensors"], 
            "bbox_coords": image_target_info["bbox_coords"],
            "bbox_positive_maps": image_target_info["bbox_positive_maps"], 
            "bbox_token_postives": image_target_info["bbox_token_postives"],
            "phrase_ids": image_target_info["phrase_ids"]
        }


class RefGameDataset(VLGDataset):
    def __init__(self, *args, split="", dstype="f30k_refgame", _configs: dict = {}, **kwargs):
        names = super().__init__(*args, split=split, dstype=dstype, _configs=_configs, **kwargs)
        self.is_training = (split == "train") 
        self.data_dir = args[0]
        self.transform_keys = args[1]
        self.use_sdr_format = _configs["use_touchdown_loss_for_cls"]
        self.use_segmentation_mask = False
        self.dstype = dstype
        self.num_distractors = _configs["refgame_options"]["num_distractors"]
        self.train_fix_contexts = _configs["refgame_options"]["train_fix_contexts"]
        self.train_fix_images = _configs["refgame_options"]["train_fix_images"]
        sampling_method_keys = {"train": "train_sampling_method", "val": "test_sampling_method", "test": "test_sampling_method"}
        self.sampling_method = _configs["refgame_options"][sampling_method_keys[split]]

        # initialize distractors
        self.initialize_distractors(names)

        # createing a gaussian kernel
        if self.use_sdr_format:
            self.initialize_gaussian_target(_configs)

        # assertion
        self.assert_dataset_options()

    def assert_dataset_options(self):
        if self.dstype == "f30k_refgame":
            if self.train_fix_contexts:
                raise NotImplementedError
            if self.train_fix_images:
                raise NotImplementedError

    def initialize_gaussian_target(self, _configs):
        self.gaussian_targets = {}
        num_items = _configs["refgame_options"]["num_distractors"] + 1
        img_size = (_configs["image_size"] * num_items, _configs["image_size"])
        scale = 1. / num_items
        for index in tqdm(range(num_items)):
            target_coords =  {
                "x": 0.5,
                "y": index * scale + scale / 2.,
            }
            target_tensor = create_gaussian_target(target_coords, img_size=img_size)
            self.gaussian_targets[index] = target_tensor
    
    def initialize_distractors(self, names: List):
        pass

    def select_distractors(self, identifier: str, index: int):
        pass

    def process_target_info(self, target_info, target_index, tokenized):
        # create target tensor
        target_tensor = self.gaussian_targets[target_index]

        # create positive map for touchdown mentions
        tokens_positive = target_info["tokens_positive"]
        positive_map = create_positive_map(tokenized, [tokens_positive], self.max_text_len)
        positive_map = torch.sum(positive_map, 0).bool()
        positive_map_list = positive_map.tolist()
        tokenized["positive_map"] = positive_map_list
        #print(self.tokenizer.decode(np.array(tokenized["input_ids"])[tokenized["positive_map"]]))
        return target_tensor, tokenized

    def get_input_and_target(self, index: int, indicies: List[int], shuffle_images: bool = True):
        """
        Create a super image / concat bounding box information / process bounding box information. 
        """

        # crate super images and apply image augumentaion?
        batch_info = {
            "images": [],
            "image_path": [],
            "boxes": [],
            "polygons": [],
            "bbox_info": [],
            "text": [],
        }
        for idx in indicies:
            image_info = self.get_raw_image(idx, image_key="image")
            bbox_info = literal_eval(str(self.table["bbox_info"][idx]))

            # apply image augumentation
            image_tensor, transformed = self.apply_image_transformations(image_info, bbox_info)
            batch_info["images"].append(image_tensor)
            batch_info["image_path"].append(image_info["image_path"])
            if idx == index:
                batch_info["boxes"] = transformed["boxes"]
                if self.use_segmentation_mask:
                    batch_info["polygons"] = transformed["polygons"]
                batch_info["bbox_info"] = bbox_info

        ## shuffle texts and images
        image_orders = [i for i in range(len(indicies))]
        if shuffle_images:
            random.shuffle(image_orders) 
            image_orders = np.array(image_orders)
            label_ind = np.argwhere(np.array(indicies) == index)[0][0]
            label_t = np.argwhere(image_orders == label_ind)[0][0]
        else:
            label_t = np.argwhere(indicies == index)[0][0]
        
        # merge image / bounding box / text
        ## create super image (stack horizontally)
        merged_image = torch.cat([batch_info["images"][o] for o in image_orders], 1)
        merged_image_path = [batch_info["image_path"][o] for o in image_orders]
                
        # process boxes and polygons
        bboxes = batch_info["boxes"]
        item_size = len(batch_info["images"])
        scale =  1. / item_size
        offset = label_t * scale
        if len(bboxes) != 0:
            bboxes[:, 1] = offset + (bboxes[:, 1] * scale)
            bboxes[:, 3] *= scale
            batch_info["boxes"] = bboxes
        if self.use_segmentation_mask:
            polygons = batch_info["polygons"]
            for i in range(len(polygons)):
                for j in range(len(polygons[i])):
                    if j % 2 == 1:
                        polygons[i][j] = offset + (polygons[i][j]  * scale)
            if len(polygons) != 0:
                batch_info["polygons"] = polygons
        
        # merge texts
        text_info = self.get_text(index)["text"]
        text, tokenized = text_info

        # bounding box target
        kwargs = {
            "image_tensor": merged_image,
            "bbox_info": batch_info["bbox_info"],
            "transformed_bboxes": batch_info["boxes"],
            "tokenized": tokenized,
            "debug": False,
            "debug_args": {"index": index, "caption": text, "image_only": False},
        } 
        if self.use_segmentation_mask:
            kwargs["transformed_polygons"] = polygons
            bbox_tensors, bbox_coords, polygon_coords, positive_maps, tokens_positives, phrase_ids = self.process_bbox_info(**kwargs)
        else:
            bbox_tensors, bbox_coords, positive_maps, tokens_positives, phrase_ids = self.process_bbox_info(**kwargs)
        
        batch = {
            "image": [merged_image],
            "image_path": merged_image_path,
            "image_orig_size": tuple(merged_image.shape[1:]), 
            "text": text_info,
            "bbox_tensors": bbox_tensors, 
            "bbox_coords": bbox_coords,
            "bbox_positive_maps": positive_maps,
            "bbox_token_postives": tokens_positives,
            "phrase_ids": phrase_ids,
            "label_t": label_t,
            "image_orders": list(image_orders)
        }
        if self.use_segmentation_mask:
            batch["polygon_coords"] = polygon_coords

        # create CLS coordinates and CLS mask 
        if self.use_sdr_format:
            target_info = {
                "tokens_positive": [[0, len(text)]], 
                "img_size": (merged_image.shape[1], merged_image.shape[2]) 
            }
            target_coords = {
                "x": 0.5,
                "y": offset + scale / 2.,
            }
            target_tensor, tokenized = self.process_target_info(target_info, label_t, tokenized)
            batch["target_tensor"] = target_tensor
            batch["target_coords"] = torch.tensor([target_coords["x"], target_coords["y"]])

        return batch
    
    def get_game_info(self, index):
        identifier = str(self.table["identifier"][index])
        if self.is_training:
            shuffle_images = True
            if not self.train_fix_images:
                context_inds = self.select_distractors(identifier, index)
                game_inds = [index] + context_inds
            else:
                game_inds = np.array(self.image_inds[index])
                shuffle_images = False
        else:
            game_inds = np.array(self.games[identifier])
            shuffle_images = False
        return {
            "indicies": game_inds,
            "shuffle_images": shuffle_images
        }

    def __getitem__(self, index):
        identifier = str(self.table["identifier"][index])
        split = str(self.table["split"][index])   
        game_info = self.get_game_info(index)
        input_target_info = self.get_input_and_target(index, indicies=game_info["indicies"], shuffle_images=game_info["shuffle_images"])

        batch = {
            "identifier": identifier,
            "split": split,
            "image": input_target_info["image"],
            "picture_path": input_target_info["image_path"],
            "picture_orig_size": input_target_info["image_orig_size"],
            "text": input_target_info["text"],
            "bbox_tensors": input_target_info["bbox_tensors"], 
            "bbox_coords": input_target_info["bbox_coords"],
            "bbox_positive_maps": input_target_info["bbox_positive_maps"], 
            "bbox_token_postives": input_target_info["bbox_token_postives"],
            "phrase_ids": input_target_info["phrase_ids"],
            "label_t": input_target_info["label_t"],
            "label_i_orders": input_target_info["image_orders"]
        }
        
        if self.use_segmentation_mask:
            batch["polygon_coords"] = input_target_info["polygon_coords"] 
        if self.use_sdr_format:
            batch["target"] = input_target_info["target_tensor"]
            batch["target_coords"] = input_target_info["target_coords"]
        
        return batch


class F30kRefGameDataset(RefGameDataset):
    def __init__(self, *args, split="", _configs: dict = {}, **kwargs):
        # initializing super class
        super().__init__(*args, split=split, dstype="f30k_refgame",_configs=_configs, **kwargs)
        self.topk = _configs["refgame_options"]["topk"]

    def initialize_distractors(self, names): 
        if self.sampling_method == "random":
            self.distractor_mapper =  {}
        else:
            clip_dir = os.path.join(self.data_dir , "clip")  
            _identifier_indicies = {str(self.table["identifier"][i]): i for i in range(len(self.table["identifier"]))}
            _imageid_indicies = {str(os.path.basename(str(self.table["image_path"][i]))): i for i in range(len(self.table["image_path"]))}

            game_dicts = {}
            for n in names:
                games_dict_path = "{}/{}.pkl".format(clip_dir, n)
                game_dicts.update(pickle.load(open(games_dict_path, "rb")))

            if self.is_training:
                self.distractor_mapper = {}
                for identifier in game_dicts.keys():
                    d_ids = game_dicts[identifier]["distractor_images"]
                    self.distractor_mapper[identifier] = [_imageid_indicies[d_id] for d_id in d_ids if d_id in _imageid_indicies]
            else:
                self.games = {}
                for identifier in game_dicts.keys():
                    curr_index = _identifier_indicies[identifier]
                    curr_image_name = os.path.basename(str(self.table["image_path"][curr_index]))
                    image_names = game_dicts[identifier]["game"]
                    self.games[identifier] =  [curr_index if img_n == curr_image_name else _imageid_indicies[img_n] for img_n in image_names]

    def select_distractors(self, identifier: str, index: int):
        """
        Sample context indicies based on CLIP ranking
        """
        distractor_indicies, context_imgnames = [], set()
        img_name = str(self.table["image_path"][index])
        context_imgnames.add(img_name)

        if self.sampling_method == "random":
            raise NotImplementedError
        elif self.sampling_method == "sample_topk":
            dis_ptr = self.topk
        elif self.sampling_method == "top_k":
            dis_ptr = self.num_distractors

        candidates = [i for i in range(len(self.table["identifier"])) if i != index] if self.sampling_method == "random" else self.distractor_mapper[identifier][:dis_ptr]

        distractor_indicies = random.sample(candidates, self.num_distractors)
        return distractor_indicies 


class TangramRefGameDataset(RefGameDataset):
    def __init__(self, *args, split="", _configs: dict = {}, **kwargs):
        # initializing super class
        super().__init__(*args, split=split, dstype="tangram_refgame", _configs=_configs, **kwargs)
        self.use_segmentation_mask = _configs["tangram_options"]["use_segmentation_mask"]
    
    def initialize_distractors(self, names):
        # load distractor candidates
        if self.is_training:
            # training
            self.distractor_mapper = {} # identifier to indicies on the table.
            self.partcount_mapper = {}
            for i in range(len(self.table["num_parts"])):
                num_part = self.table["num_parts"][i]
                self.partcount_mapper.setdefault(num_part, [])
                self.partcount_mapper[num_part].append(i)

            # debug context calculation
            runout_contexts = False
            for num_part in self.partcount_mapper.keys():
                if len(self.partcount_mapper[num_part]) < self.num_distractors:
                    runout_contexts = True
                    break

            if runout_contexts:
                self.partcount_mapper = {}
                data_len = len(self.table["identifier"])
                for i in range(data_len):
                    identifier = str(self.table["identifier"][i])
                    self.distractor_mapper[identifier] = [j for j in range(data_len) if j != i]
            
            # use fixed set of distractors and images during training
            if self.train_fix_contexts:
                distractor_mappers = {}
                if self.train_fix_images:
                    image_inds = {}

                data_len = len(self.table["identifier"])
                for index in range(data_len):
                    identifier = str(self.table["identifier"][index])
                    if identifier in self.distractor_mapper:
                        # if any pre-defined contexts exist
                        distractor_mappers[identifier] = random.sample(self.distractor_mapper[identifier], self.num_distractors)
                    else:
                        # otherwise, sample distractors
                        distractor_mappers[identifier] = self.select_distractors(identifier, index)

                    # fix image orders
                    if self.train_fix_images:
                        image_order = [index] + distractor_mappers[identifier] 
                        random.shuffle(image_order) 
                        image_inds[index] = image_order

                self.distractor_mapper = distractor_mappers
                if self.train_fix_images:
                    self.image_inds = image_inds
        else:
            # loading pre-calculated refernce games
            game_dicts = {}
            for n in names:
                games_dict_path = "{}/{}.pkl".format(self.data_dir, n.replace(".arrow", ""))
                assert os.path.exists(games_dict_path), "make sure tangram refernce game is pre-calculated and is located under the dataset folder."
                game_dicts.update(pickle.load(open(games_dict_path, "rb")))
            
            # FIXME move phrase_id processing from evaluator to here
            # repeat and populate the table
            new_table = None
            batches = []
            columns = self.table.column_names
            for i in range(len(self.table)):
                identifier = str(self.table["identifier"][i])
                if identifier not in game_dicts:
                    print("{} is missing from a batch file.".format(identifier))
                else:
                    num_repeat = len(game_dicts[identifier])
                    for j in range(num_repeat):
                        batch = {}
                        for col in columns:
                            batch[col] = self.table[col][i].as_py()
                            if col == "identifier":
                                batch[col] = str(self.table[col][i]) + "_{}".format(j)
                        batches.append(batch)
            dataframe = pd.DataFrame(
                batches, columns=columns,
            )
            new_table = pa.Table.from_pandas(dataframe)
            self.table = new_table

            # calculating validation games
            self.games = {} # identifier ==> indicies on table
            _identifier_indicies = {str(self.table["identifier"][i]): i for i in range(len(self.table["identifier"]))}
            for id in game_dicts.keys():
                ctx_ids = game_dicts[id]
                for did in range(len(ctx_ids)):
                    dup_id = str(id) + f"_{did}"
                    self.games[dup_id] = [_identifier_indicies[dup_id] if ctx_id == id else _identifier_indicies[ctx_id + "_0"] for ctx_id in ctx_ids[did][:self.num_distractors+1]]
            
    def select_distractors(self, identifier: str, index: int):
        """
        Sample context indicies based on CLIP ranking
        """
        distractor_indicies = []
        used_names, used_text = set(), set()
        if len(self.distractor_mapper) != 0:
            candidates = self.distractor_mapper[identifier]
        else:
            num_part = self.table["num_parts"][index]
            candidates = set(self.partcount_mapper[num_part])
            candidates.remove(index)

        if self.is_training:
            while True:
                candidates = copy.deepcopy(candidates)
                distractor_index = random.sample(candidates, 1)[0]
                candidates.remove(distractor_index)
                tangram_name = str(self.table["image_path"][distractor_index])
                if "_" in tangram_name:
                    tangram_name = "_".join(os.path.basename(tangram_name).split("_")[:-1])
                text = str(self.table["caption"][distractor_index])
                '''
                [1] no 2 annotations are for the same tangram
                [2] no 2 annotations are the same 
                '''
                if (tangram_name not in used_names) and (text not in used_text):
                    distractor_indicies.append(distractor_index)    
                    used_names.add(tangram_name)
                    used_text.add(text)

                if len(distractor_indicies) == self.num_distractors:
                    break
            return distractor_indicies 
        else:
            return candidates

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        img_name = str(self.table["image_path"][index])
        batch["tangram_id"] = os.path.basename(img_name).replace(".png", "").replace(".jpg", "")
        return batch
