# python -m src.preprocessing.write_coco_format_datasets
from __future__ import from __future__ import annotations

import argparse
import json
import os
import copy
from collections import defaultdict
from typing import List, Dict, Union, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import cv2
from tqdm import tqdm
import torch
from pycocotools import mask as coco_mask

def convert_abs_rel(bbox: np.array, img_size: Tuple[int, int]) -> List[float]:
    """
    Convert box in absolute coordinates to relative coordinates.
    """
    h, w = img_size
    bbox = np.array(bbox, dtype=float)
    bbox[[0, 2]] /= w
    bbox[[1, 3]] /= h
    return list(bbox)

def convert_abs_rel_segmentation(polygons: np.array, img_size: Tuple[int, int]) -> np.array:
    h, w = img_size
    for polygon in polygons:
        for j in range(len(polygon)):
            if j % 2 == 0:
                polygon[j] /= w
            else:
                polygon[j] /= h
    return polygons


def convert_rel_abs_segmentation(polygons: np.array, img_size: Tuple[int, int]) -> np.array:
    h, w = img_size
    for polygon in polygons:
        for j in range(len(polygon)):
            if j % 2 == 0:
                polygon[j] *= w
            else:
                polygon[j] *= h
    return polygons


def convert_coco_poly_to_mask(segmentations: List[np.array], height: int, width: int) -> torch.Tensor:
    masks = []
    for polygons in segmentations:
        convert_rel_abs_segmentation(polygons, (height, width))
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


def save_table(batches: List[Dict], file_path: str, columns: List[str] = None) -> None:
    if columns is None:
        columns = ["identifier", "image_path", "image", "caption", "bbox_info", "split"]
        
    dataframe = pd.DataFrame(batches, columns=columns)
    table = pa.Table.from_pandas(dataframe)
    
    with pa.OSFile(file_path, "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)


def format_coco_annotations(js: Dict, split: str, image_folder: str, output_folder: str = "", debug: bool = False, corr_annt_prop: float = 1.0, num_debug_examples: int = -1, is_toucdown_sdr: bool = False, is_phrasecut: bool = False, is_tangram: bool = False, verbose: bool = True, precalculate_images: bool = False, visualize_annt: bool = False) -> list:
    data = {}

    # If debugging, shuffle and limit the images
    if debug:
        if not is_toucdown_sdr:
            random.shuffle(js["images"])
        js["images"] = js["images"][:num_debug_examples]

    # Process images
    for j in tqdm(js["images"]):
        ds_name = j["data_source"] if "data_source" in j else j["dataset_name"]

        if type(image_folder) == dict:
            if ds_name == "coco":
                ds_image_folder = image_folder["coco_img_dir"]
            elif "refcoco" in ds_name:
                ds_image_folder = image_folder["coco_img_dir"]
                ds_name = "refcoco" 
            elif ds_name == "vg" or ds_name == "gqa":
                ds_image_folder = image_folder["vg_img_dir"]
            else:
                raise ValueError(f"Dataset: {ds_name} is unexpected.")
        
            image_path = "{}/{}".format(ds_image_folder, j["file_name"])
        else:
            assert(ds_name == "flickr" or ds_name == "touchdown_sdr" or ds_name == "phrasecut" or ds_name == "tangram")
            image_path = "{}/{}".format(image_folder, j["file_name"])

        identifier = "{}_{}_{}".format(ds_name, split, j["id"])
        raw_identifier = j["id"]
        data[raw_identifier] = {
            "identifier": identifier,
            "image_path": image_path,
            "caption": j["caption"],
            "split": f"{ds_name}_{split}" if ds_name != "touchdown_sdr" else split,
            "img_size": (int(j["height"]), int(j["width"])),
            "bbox_info_temp": {} 
        }

        # Touchdown SDR 
        if is_toucdown_sdr:
            pattern = re.compile("touchdown|bear", re.IGNORECASE)
            tokens_positive = []
            for match in re.finditer(pattern, j["caption"]):
                span = match.span()
                tokens_positive.append(list(span))

            sdr_info = {
                "tokens_positive": tokens_positive,
                "coords": j["sdr_target_coords"],
            }
            data[raw_identifier].update({
                "sdr_info": sdr_info,
                "caption_id": j["caption_id"], # necessary for consistency evaluation
                "is_propagated": j["is_propagated"]   # relative directions in propagated images are possibly corrupted.
            })

        # phrasecut 
        if is_phrasecut:
            data[raw_identifier].update({
                "original_id": j["original_id"],
                "task_id": j["task_id"]
            })

        # tangram
        if is_tangram:
            data[raw_identifier]["num_parts"] = j["num_parts"]

        # encoding images
        if precalculate_images:
            with open(image_path, "rb") as fp:
                binary_image = fp.read()
                data[raw_identifier]["image"] = binary_image
        else:
            assert(os.path.exists(image_path))
            data[raw_identifier]["image"] = None
    
    # formatting regions-to-phrases annotations
    phrase_ct = 0
    removed_phrases = 0
    for j in tqdm(js["annotations"]):
        raw_identifier = j["image_id"]

        if raw_identifier not in data:
            removed_phrases += 1
            continue

        # assigning the phrase id
        identifier = data[raw_identifier]["identifier"]
        ds_name = identifier.split("_")[0]
        phrase_id = f"{ds_name}_{split}_{phrase_ct}"
        
        # set up a bbox info dictionary
        tokens_positive = str(j["tokens_positive"]) 
        if tokens_positive not in data[raw_identifier]["bbox_info_temp"]:
            phrase_ct += 1
        data[raw_identifier]["bbox_info_temp"].setdefault(tokens_positive, {"phrase_id": phrase_id, "bboxes": [], "segmentation": []})

        # adding bounding box to the phrase
        rel_bbox = convert_abs_rel(j["bbox"], data[raw_identifier]["img_size"]) # [x, y, w, h]
        data[raw_identifier]["bbox_info_temp"][tokens_positive]["bboxes"].append(rel_bbox)
        if "segmentation" in j:
            data[raw_identifier]["bbox_info_temp"][tokens_positive]["segmentation"] += convert_abs_rel_segmentation(j["segmentation"], data[raw_identifier]["img_size"])
    
    # reformatting bbox info so that the phrase id will be the key
    all_phrases = set()
    for raw_identifier in tqdm(data.keys()):
        data[raw_identifier]["bbox_info"] = {}
        for k, v in data[raw_identifier]["bbox_info_temp"].items():
            all_phrases.add(v["phrase_id"])
            if "phrasecut" in data[raw_identifier]["split"]:
                assert(len(v["segmentation"]) > 0)

            data[raw_identifier]["bbox_info"][v["phrase_id"]] = {
                "tokens_positive": literal_eval(k),
                "bboxes": v["bboxes"],
                "segmentation": v["segmentation"]
            }

    # reformatting assertion
    phrase_ct_assert = 0
    for raw_identifier in data.keys():
        phrase_ct_assert += len(data[raw_identifier]["bbox_info"])
    assert(phrase_ct == phrase_ct_assert)

    # sampling
    sampled_phrases = sample(all_phrases, int(len(all_phrases)* corr_annt_prop))
    for raw_identifier in data.keys():
        pids = list(data[raw_identifier]["bbox_info"].keys())
        for pid in pids:
            if pid not in sampled_phrases:
                del data[raw_identifier]["bbox_info"][pid]

    phrase_ct = 0
    for raw_identifier in data.keys():
        phrase_ct += len(data[raw_identifier]["bbox_info"])

    if visualize_annt:
        for raw_identifier in tqdm(data.keys()):
            identifier = data[raw_identifier]["identifier"]
            ds_name = identifier.split("_")[0]

            # create output folder
            folder_path = f"{output_folder}/{ds_name}/visualization"
            os.makedirs(folder_path, exist_ok=True)

            bbox_info = data[raw_identifier]["bbox_info"]

            for phrase_id in bbox_info:
                out_image_path = f"{folder_path}/{raw_identifier}_{phrase_id}.jpg"
                img = cv2.imread(data[raw_identifier]["image_path"])
                hs, ws, _  = img.shape

                # annotate bounding box
                for b in bbox_info[phrase_id]["bboxes"]:
                    b = np.array(b)
                    b[[0,2]] *= ws
                    b[[1,3]] *= hs
                    x, y, w, h = b.astype(int)
                    img = cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 1)
                # annotate captions
                cap = " ".join([data[raw_identifier]["caption"][tk[0]:tk[1]] for tk in bbox_info[phrase_id]["tokens_positive"]])
                cv2.putText(img, cap, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

                # annotate sgemeentation
                mask = convert_coco_poly_to_mask([bbox_info[phrase_id]["segmentation"]], hs, ws)
                mask = mask.numpy() 
                mask *= 255
                mask = cv2.cvtColor(mask[0, ...], cv2.COLOR_GRAY2BGR)
                mask = cv2.convertScaleAbs(mask)
                mask_img = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                from IPython import embed
                img = cv2.addWeighted(mask_img, 0.3, img, 0.7, 0)
                cv2.imwrite(out_image_path, img)
                print(out_image_path)
    
    batches = []
    num_bboxes = []
    num_discarded_images = 0
      
    for v in data.values():
        if len(v["bbox_info"].values()) == 0 and (not is_toucdown_sdr and not is_tangram):
            num_discarded_images += 1
            continue

        for p in v["bbox_info"].values():
            num_bboxes.append(len(p["bboxes"]))

        batch =  [
            v["identifier"],
            v["image_path"],
            v["image"],
            v["caption"],
            str(v["bbox_info"]),
            v["split"],
        ]

        # Touchdown SDR specific  
        if is_toucdown_sdr:
            batch += [
                str(v["sdr_info"]), 
                v["caption_id"], 
                v["is_propagated"]
            ]

        if is_phrasecut:
            batch += [
                v["original_id"], 
                v["task_id"]
            ]
        
        if is_tangram:
            batch += [
                v["num_parts"]
            ]

        batches.append(batch)

    if verbose:
        num_bboxes = np.array(num_bboxes)
        print(" ({}): {} examples ({} discarded), {} phrases ({} phrases removed, {} phrases with multiple bounding boxes, avg {} bboxes per phrase)).".format(split, len(batches), num_discarded_images, phrase_ct, removed_phrases, np.sum(num_bboxes != 1) , np.mean(num_bboxes)))

    return batches

def load_flickr_annotations(annotation_path: str):
    annts = {}
    annts["train"] = json.load(open(os.path.join(annotation_path, "final_flickr_separateGT_train.json"), "r"))
    annts["dev"] = json.load(open(os.path.join(annotation_path, "final_flickr_separateGT_val.json"), "r"))
    annts["test"] = json.load(open(os.path.join(annotation_path, "final_flickr_separateGT_test.json"), "r"))
    return annts


def load_touchdown_sdr_annotations(annotation_path: str):
    annts = {}
    annts["train"] = json.load(open(os.path.join(annotation_path, "finetune_touchdown_sdr_train_800_3712_0.5_True_True.json"), "r"))
    annts["tune"] = json.load(open(os.path.join(annotation_path, "finetune_touchdown_sdr_tune_800_3712_0.5_True_True.json"), "r"))
    annts["dev"] = json.load(open(os.path.join(annotation_path, "finetune_touchdown_sdr_dev_800_3712_0.5_True_True.json"), "r"))
    annts["test"] = json.load(open(os.path.join(annotation_path, "finetune_touchdown_sdr_test_800_3712_0.5_True_True.json"), "r"))
    annts["debug"] = json.load(open(os.path.join(annotation_path, "finetune_touchdown_sdr_debug_800_3712_0.5_True_True.json"), "r"))
    return annts


def load_tangram_annotations(annotation_path: str):
    annts = {}
    annts["train"] = json.load(open(os.path.join(annotation_path, "finetune_tangram_train.json"), "r"))
    annts["val"] = json.load(open(os.path.join(annotation_path, "finetune_tangram_val.json"), "r"))
    annts["dev"] = json.load(open(os.path.join(annotation_path, "finetune_tangram_dev.json"), "r"))
    annts["test"] = json.load(open(os.path.join(annotation_path, "finetune_tangram_test.json"), "r"))
    annts["debug"] = json.load(open(os.path.join(annotation_path, "finetune_tangram_val.json"), "r"))
    return annts

def make_flickr_arrow(image_folder: str, annotation_path: str, output_folder: str, debug: bool = False, num_debug_examples: int = 16):
    # create output dir
    os.makedirs(output_folder, exist_ok=True)

    # load flickt annotation
    annts = load_flickr_annotations(annotation_path)

    # process sdr / bbox annotations
    if debug:
        input_splits = ["train", "train", "dev"]
        output_splits = ["train", "dev", "test"]
    else:
        input_splits = ["train", "dev", "test"]
        output_splits = ["train", "dev", "test"]

    for in_split, out_split in zip(input_splits, output_splits):
        flickr_batches = format_coco_annotations(annts[in_split], out_split, image_folder, debug=debug, num_debug_examples=num_debug_examples)

        print("{} examples: {} ==> {}".format(len(flickr_batches), in_split, out_split))

        save_table(flickr_batches, f"{output_folder}/flickr30k_entities_{out_split}.arrow")


def make_touchdown_sdr_arrow(image_folder: str, annotation_path: str, output_folder: str, debug: bool = False, num_debug_examples: int = 16, corr_annt_prop: float = 1.0):
    # create output dir
    os.makedirs(output_folder, exist_ok=True)

    # load flickt annotation
    annts = load_touchdown_sdr_annotations(annotation_path)

    # process sdr / bbox annotations
    if debug:
        input_splits = ["debug", "debug", "tune"]
        output_splits = ["train", "tune", "dev"]
    else:
        input_splits = ["train", "tune", "dev", "test"]
        output_splits = ["train", "tune", "dev", "test"]
        """
        input_splits = ["dev"]
        output_splits = ["dev"]
        """


    for in_split, out_split in zip(input_splits, output_splits):
        sdr_batches = format_coco_annotations(annts[in_split], out_split, image_folder, debug=debug, num_debug_examples=num_debug_examples, is_toucdown_sdr=True, precalculate_images=True, corr_annt_prop=corr_annt_prop if out_split == "train" else 1.0)

        print("{} examples: {} ==> {}".format(len(sdr_batches), in_split, out_split))
        save_table(sdr_batches, f"{output_folder}/flickr30k_entities_{out_split}.arrow", columns=["identifier", "image_path", "image", "caption", "bbox_info", "split", "sdr_info", "caption_id", "is_propagated"])


def make_tangram_arrow(image_folders: Dict, annotation_path: str, output_folder: str, debug: bool = False, num_debug_examples: int = 16):
    # create output dir
    os.makedirs(output_folder, exist_ok=True)

    # load flickt annotation
    annts = load_tangram_annotations(annotation_path)

    # process sdr / bbox annotations
    if debug:
        input_splits = ["debug", "debug", "val", "test"]
        output_splits = ["train", "val", "dev", "test"]
    else:
        input_splits = ["train", "val", "dev", "test"]
        output_splits = ["train", "val", "dev", "test"]

    for in_split, out_split in zip(input_splits, output_splits):
        image_folder = image_folders[in_split]
        tangram_batches = format_coco_annotations(annts[in_split], out_split, image_folder, output_folder, debug=debug, num_debug_examples=num_debug_examples, is_tangram=True,visualize_annt=False)
        print("{} examples: {} ==> {}".format(len(tangram_batches), in_split, out_split))

        save_table(tangram_batches, f"{output_folder}/tangram_{out_split}.arrow", columns=["identifier", "image_path", "image", "caption", "bbox_info", "split", "num_parts"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run different dataset modes.")
    
    parser.add_argument("--mode", choices=["touchdown_sdr", "flickr", "tangram"], help="Select mode to run.")
    parser.add_argument('--image_dir', type=str, default='', help='the directory containing images')
    parser.add_argument('--annotation_dir', type=str, default='', help='the directory containing images')
    parser.add_argument("--debug", action="store_true", help="Run in debug mode if set.")

    args = parser.parse_args()

    if args.mode == "touchdown_sdr":
        make_touchdown_sdr_arrow(
            image_folder=args.image_dir,
            annotation_path=args.annotation_dir,
            output_folder="data/touchdown_sdr_cocoformat/800_3712/debug" if args.debug else "data/touchdown_sdr_cocoformat/800_3712/native",
            debug=args.debug,
            num_debug_examples=16
        )
    elif args.mode == "flickr":
        make_flickr_arrow(
            image_folder=args.image_dir,
            annotation_path=args.annotation_dir,
            output_folder="data/f30k_ref/debug" if args.debug else "data/f30k_ref/native",
            debug=args.debug
        )
    elif args.mode == "tangram":
        image_folders_config = {
            "train": "{}/train/images".format(image_folder=args.image_dir),
            "debug": "{}/val/images".format(image_folder=args.image_dir),
            "val": "{}/val/images".format(image_folder=args.image_dir),
            "dev": "{}/dev/images".format(image_folder=args.image_dir),
            "test": "{}/heldout/images/color".format(image_folder=args.image_dir),
        }
        make_tangram_arrow(
            image_folders=image_folders_config,
            annotation_path=args.annotation_dir,
            output_folder="data/tangram/debug" if args.debug else "data/tangram/native",
            debug=args.debug
        )

