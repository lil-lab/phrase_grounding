# python -m src.preprocessing.preprocess_tangram
import os
import re
import cv2
import random
import json
import inflect
import itertools
import numpy as np
from tqdm import tqdm
from IPython import embed
from nltk.corpus import wordnet as wn

p = inflect.engine()

svg_folder_path = "/home/nk654/projects/p-tangram/tangrams-dev/CLIP/make-data/make-png/tangrams-svg"

train_text_path = "./data/tangram/train/train_part_sw.json"
train_data_path = "./data/tangram/train/color_texts_train.json"
train_image_path = "./data/tangram/train/images"

aug_train_text_path = "/home/nk654/projects/p-tangram/tangrams-dev/CLIP/fine-tuning/data/new-augmented-annotations/train_part_sw.json"
aug_train_data_path = "/home/nk654/projects/p-tangram/tangram_data/augmented/texts/new_powerset_color_texts_train.json"
aug_train_image_path = "/home/nk654/projects/p-tangram/tangram_data/augmented/images/new_powerset_png_train"

test_text_path = "./data/tangram/heldout/heldout_part_sw.json"
test_data_path = "./data/tangram/heldout/color_texts_heldout.json"
test_image_path = "./data/tangram/heldout/images/color"

dev_text_path = "./data/tangram/dev/dev_part_sw.json"
dev_data_path = "./data/tangram/dev/color_texts_dev.json"
dev_image_path = "./data/tangram/dev/images"

val_text_path = "./data/tangram/val/val_part_sw.json"
val_data_path = "./data/tangram/val/color_texts_val.json"
val_image_path = "./data/tangram/val/images"



IMG_SIZE = 224

import torch
from pycocotools import mask as coco_mask
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


def process_data(text_path: str, data_path: str, image_folder: str, svg_folder: str, visualize_grounding_annotation: bool = False):
    # processing data files
    json_data = json.load(open(text_path))
    image_names, tangrams_names, texts, parts_counts = [],[],[],[] 
    for tangrams_name, annts in json_data.items():
        for i, ann in enumerate(annts):
            image_name = tangrams_name + '_' + str(i) + '.png'
            pc = ann.count('#') #dog#head#tail has 2 parts
            image_names.append(image_name) #['page1-1_2.png', ...]
            tangrams_names.append(tangrams_name) # [page1-1,...]
            texts.append(ann) #[dog#head#tail, ...]
            parts_counts.append(pc) #[2, ...]

    # get a segmentation and bounidng box information
    raw_data = json.load(open(data_path))
    pattern = r'id="(.*?)"'
    baseinfo_pattern ='viewBox="[\w.,\-\s]+"'
    polygon_pattern = 'points="[\w.\-,\s]+"'
    imagename_to_phrasepolygon = {}
    imagename_to_phrasebox = {}

    total_ct = 0

    for tangrams_name in tqdm(set(tangrams_names)):
        svg_path = os.path.join(svg_folder, tangrams_name + ".svg")
        f = open(svg_path,'r').read().replace("\n"," ")
        polygons = f.split('polygon')
        data = raw_data[tangrams_name]
        total_ct+= len(data)

        for d_ind, d in enumerate(data):
            phraseid_to_polygon = {}
            phraseid_to_box = {}
            pieceid_to_phraseid = {}
            for p_id, c in enumerate(d["color_groups"]):
                for piece_id in c:
                    pieceid_to_phraseid[piece_id] = p_id

            if len(d["color_groups"]) != 0:
                base_info = re.findall(baseinfo_pattern, polygons[0])[0] #piece id
                base_info = base_info.replace("viewBox=", "")
                base_info = base_info.replace(r'"', '')
                min_x, min_y, width, height = base_info.split(" ")
                min_x, min_y, width, height = float(min_x), float(min_y), float(width), float(height)
                img_size = max([width, height])
                if height > width:
                    min_x = (width - height) / 2
                elif height < width:
                    min_y = (height - width) / 2

                for s in polygons[1:]: 
                    polygon_info = re.findall(polygon_pattern, s)[0] #piece id
                    polygon_info = polygon_info.replace("points=", "")
                    polygon_info = polygon_info.replace(r'"', '')
                    polygon_info = polygon_info.strip()
                    if "," not in polygon_info:
                        polygon = []
                        coords = polygon_info.split(" ")
                        for c_ind in range(int(len(coords) / 2)):
                            polygon.append([np.clip((float(coords[c_ind])-min_x) / img_size, 0, 1) * IMG_SIZE, np.clip((float(coords[c_ind+1])-min_y) / img_size, 0, 1) * IMG_SIZE])
                    else:
                        polygon = [
                            [
                                np.clip((float(p.split(",")[0])-min_x) / img_size, 0, 1) * IMG_SIZE,
                                np.clip((float(p.split(",")[1])-min_y) / img_size, 0, 1) * IMG_SIZE
                            ]
                            for p in polygon_info.split(" ")
                        ]
                    merged_polygon = list(itertools.chain(*polygon))
                    polygon = np.array(polygon)
                    x, y = np.min(polygon[:, 0]) / IMG_SIZE, np.min(polygon[:, 1]) / IMG_SIZE
                    x2, y2 = np.max(polygon[:, 0]) / IMG_SIZE, np.max(polygon[:, 1]) / IMG_SIZE
                    w, h = x2 - x, y2 - y
                    box = [x*IMG_SIZE,y*IMG_SIZE,w*IMG_SIZE,h*IMG_SIZE]
                    
                    piece_id = re.findall(pattern, s)[0] #piece id
                    if piece_id in pieceid_to_phraseid:
                        phraseid = pieceid_to_phraseid[piece_id]
                        phraseid_to_polygon.setdefault(phraseid, [])
                        phraseid_to_polygon[phraseid].append(merged_polygon)
                        phraseid_to_box.setdefault(phraseid, [])
                        phraseid_to_box[phraseid].append(box)

            image_name = tangrams_name + '_' + str(d_ind) + '.png'
            imagename_to_phrasepolygon[image_name] = phraseid_to_polygon
            imagename_to_phrasebox[image_name] = phraseid_to_box

    # verify segmentation / bounding boxes
    if visualize_grounding_annotation:
        inds = [i for i in range(len(image_names))]
        sampled_inds = random.sample(inds, 30)
        image_outpath = "./src/preprocessing/outputs" 
        os.makedirs(image_outpath, exist_ok=True)
        for id in sampled_inds:
            image_name = image_names[id]
            text = texts[id]
            phrase_to_polygon = imagename_to_phrasepolygon[image_name]
            phrase_to_bbox = imagename_to_phrasebox[image_name]
            phrases = text.split("#")[1:]

            image_path = os.path.join(image_folder, image_name)

            for pid in range(len(phrases)):
                img = cv2.imread(image_path)
                hs, ws, _  = img.shape

                # annotating bounding boxes
                bboxes = phrase_to_bbox[pid]
                for b in bboxes:
                    b = np.array(b)
                    x, y, w, h = b.astype(int)
                    img = cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 1)

                # annotating segmentation masks
                polygons = phrase_to_polygon[pid]
                mask = convert_coco_poly_to_mask([polygons], hs, ws)
                mask = mask.numpy() 
                mask *= 255
                mask = cv2.cvtColor(mask[0, ...], cv2.COLOR_GRAY2BGR)
                mask = cv2.convertScaleAbs(mask)
                mask_img = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                img = cv2.addWeighted(mask_img, 0.3, img, 0.7, 0)


                # saving image
                phrase = phrases[pid]
                out_image_path = os.path.join(image_outpath, image_name.replace(".png", "") + phrase.replace(" ", "_") + ".png")
                cv2.imwrite(out_image_path, img)

    # organize data in coco format
    coco_images, coco_annotations = [], []
    annt_id = 0
    coco_categories = [{"supercategory": "object", "id": 1, "name": "object"}]

    for i, image_name in enumerate(image_names):
        text = texts[i]
        whole_parts = text.split("#")
        num_parts = text.count("#")

        # check whole phrase is a singular and if so add a proper article if missing.
        whole_phrase = whole_parts[0]
        last_word = whole_phrase.split(" ")[-1]
        if not p.singular_noun(last_word):
            whole_phrase = p.a(whole_phrase)

        # clean part phrases (remove UNKNOWN)
        part_phrases = whole_parts[1:]
        phrases_ids, cleaned_part_phrases = [], []
        for p_id, part in enumerate(part_phrases):
            if part != "UNKNOWN":
                cleaned_part_phrases.append(part)
                phrases_ids.append(p_id)

        # create a caption by iteratively adding part phrases
        caption = "This looks like {} with ".format(whole_phrase) if len(cleaned_part_phrases) != 0 else "This looks like {}.".format(whole_phrase) 
        tokens_positive = []

        for j in range(len(cleaned_part_phrases)):
            # check part phrase is a singular and if so add a proper article if missing.
            part_phrase = cleaned_part_phrases[j]
            last_word = part_phrase.split(" ")[-1]
            if not p.singular_noun(last_word):
                part_phrase = p.a(part_phrase)

            # update a caption, tokens positive and a phrase list
            start = len(caption)
            caption += part_phrase
            end = len(caption)
            tokens_positive.append((start, end))

            if j == len(cleaned_part_phrases) - 1:
                # the last part phrase
                caption += "."
            elif j == len(cleaned_part_phrases) - 2:
                # the second last part phrase
                caption += ", and "
            else:   
                # part phrases coming in between
                caption += ", "

        cur_img = {
            "file_name": image_name,
            "height": 224,
            "width": 224,
            "id": i,
            "caption": caption,
            "num_parts": num_parts,
            "tokens_negative": [[(0, len(caption))]],
            "dataset_name": "tangram",
            "task_id": i,
        }
        coco_images.append(cur_img)


        # calculate bounding boxes and segmentation masks
        phrase_to_polygon = imagename_to_phrasepolygon[image_name]
        phrase_to_bbox = imagename_to_phrasebox[image_name]
        for c_p_id, p_id in enumerate(phrases_ids):
            polygons = phrase_to_polygon[p_id]
            boxes = phrase_to_bbox[p_id]
            if len(polygons) > 0:
                new_boxes, new_polygons = [], []
                for polygon, box in zip(polygons, boxes):
                    new_boxes.append(polygon)
                    new_polygons.append(box)

                new_polygons = []
                for polygon, box in zip(polygons, boxes):
                    x, y, w, h = box
                    #print(caption[tokens_positive[c_p_id][0]:tokens_positive[c_p_id][1]])
                    cur_obj = {
                        "area": h * w,
                        "iscrowd": 0,
                        "category_id": 1,
                        "bbox": box,
                        "segmentation": [polygon],
                        "tokens_positive": [tokens_positive[c_p_id]],
                        "image_id": i,
                        "id": annt_id,
                    }
                    annt_id += 1
                    coco_annotations.append(cur_obj)

    ds = {"info": [], "licenses": [], "images": coco_images, "annotations": coco_annotations, "categories": coco_categories}
    return ds


if __name__ == "__main__":
    out_folder = "./data/tangram/coco_jsons"
    os.makedirs(out_folder, exist_ok=True)

    # training data
    train_json = process_data(train_text_path, train_data_path, train_image_path, svg_folder_path)
    output_path = os.path.join(out_folder, "finetune_tangram_train.json")
    with open(output_path, "w") as j_file:
        json.dump(train_json, j_file)

    # training augumentated data
    train_json = process_data(aug_train_text_path, aug_train_data_path, aug_train_image_path, svg_folder_path)
    output_path = os.path.join(out_folder, "finetune_tangram_train_aug.json")
    with open(output_path, "w") as j_file:
        json.dump(train_json, j_file)

    # test data
    test_json = process_data(test_text_path, test_data_path, test_image_path, svg_folder_path)
    output_path = os.path.join(out_folder, "finetune_tangram_test.json")
    with open(output_path, "w") as j_file:
        json.dump(test_json, j_file)

    # valisation data
    val_json = process_data(val_text_path, val_data_path, val_image_path, svg_folder_path)
    output_path = os.path.join(out_folder, "finetune_tangram_val.json")
    with open(output_path, "w") as j_file:
        json.dump(val_json, j_file)

    # development data
    dev_json = process_data(dev_text_path, dev_data_path, dev_image_path, svg_folder_path)
    output_path = os.path.join(out_folder, "finetune_tangram_dev.json")
    with open(output_path, "w") as j_file:
        json.dump(dev_json, j_file)

