# python -m src.preprocessing.f30k.calculate_f30k_games
import os, pickle, copy
import random
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
from IPython import embed

import torch
import clip

num_val_distractors = 5
num_train_distractors = 20
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_clip():
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

def load_flickr_annotations(annotation_path: str):
    annts = {}
    annts["train"] = json.load(open(os.path.join(annotation_path, "final_flickr_separateGT_train.json"), "r"))
    annts["dev"] = json.load(open(os.path.join(annotation_path, "final_flickr_separateGT_val.json"), "r"))
    annts["test"] = json.load(open(os.path.join(annotation_path, "final_flickr_separateGT_test.json"), "r"))
    return annts

def reformat_annotation(js, image_folder):
    data = {}

    for j in tqdm(js["images"]):
        raw_identifier = j["id"]
        identifier = "{}_{}".format("flickr", j["id"])
        image_path = "{}/{}".format(image_folder, j["file_name"])
        data[raw_identifier] = {
            "identifier": identifier,
            "image_path": image_path,
            "caption": j["caption"].lower(), # lowering caption
        }
    
    return data

@torch.no_grad()
def calculate_games(data, model, preprocess, topk: int = 20, verbose: bool = False, out_path: str = ""):
    games = {} # each id to image
    keys = list(data.keys())
    all_images = [data[k]["image_path"] for k in keys]
    image_pathes = list(set(all_images))
    batch_size = 64

    # calculating image features
    offset = 0
    image_features_list = []
    num_iters = np.ceil(len(image_pathes) / batch_size).astype(int)

    for it in tqdm(range(num_iters)):
        endset = min(offset + batch_size, len(image_pathes))
        inds = range(offset, endset)
        images = [preprocess(Image.open(image_pathes[ind])) for ind in inds]
        images = torch.stack(images, 0).to(device)
        offset += batch_size

        # model inference
        image_features = model.encode_image(images)
        image_features_list.append(image_features)

    for k in tqdm(keys):
        caption = data[k]["caption"]
        text = clip.tokenize([caption], truncate=True).to(device)
        text_features = model.encode_text(text)

        # Pick the top k most similar labels for the image
        image_features = torch.cat(image_features_list,0)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=0)

        image_path = data[k]["image_path"]
        remove_idx = np.argwhere(np.array(image_pathes) == image_path)[0][0]
        similarity[remove_idx, 0]  = 0
        values, indices = similarity[:, 0].topk(topk)
        key = "flickr_{}".format(k)
        game = [os.path.basename(image_pathes[i]) for i in indices] + [os.path.basename(image_path)]
        random.shuffle(game)
        games[key] = {
                "distractor_images":[os.path.basename(image_pathes[i]) for i in indices],
                "similarities": values.tolist(),
                "game": game,
        }

    if verbose:
        sampled_keys = random.sample(list(games.keys()), 50)
        for k in sampled_keys:
            old_k = int(k.replace("flickr_", ""))
            caption = data[old_k]["caption"]
            image_path = data[old_k]["image_path"] 
            print("="*50)
            print("Image path: {}, Caption: {}".format(image_path, caption))
            distractor_images = games[k]["distractor_images"]
            sims = games[k]["similarities"]
            image_folder = os.path.dirname(image_path)
            for d, s in zip(distractor_images, sims):
                print(os.path.join(image_folder, d), s)

    pickle.dump(games, open(out_path, "wb"))

if __name__ == "__main__":
    # load clip componentes
    model, preprocess = load_clip()
    splits = ["train", "dev", "test"]
    out_folder = "/home/nk654/projects/p-vl-grounding/data/f30k_ref/native/clip"
    os.makedirs(out_folder, exist_ok=True)
    for sp in splits:
        # load flickr images
        js = load_flickr_annotations(
            annotation_path="/home/nk654/projects/p-interactive-touchdown/data/coco_format/"
        )
        data = reformat_annotation(
            js[sp], 
            image_folder="/home/nk654/projects/p-interactive-touchdown/data/flickr30k/flickr30k-images/all",
        )

        # caluclate clip score and rank image / text pairs
        out_path = os.path.join(out_folder, "flickr30k_entities_{}.pkl".format(sp))
        topk = num_train_distractors if sp == "train" else num_val_distractors
        calculate_games(data, model, preprocess, topk=topk, out_path=out_path)


"""
if __name__ == "__main__":
    folder_path = "data/tangram/single_batch" 

    # load data file
    for debug in (True,):
        folder_path = "/home/nk654/projects/p-interactive-touchdown/models/Touchdown-ViLT/data/f30k_ref/debug/clip" if debug else "/home/nk654/projects/p-interactive-touchdown/models/Touchdown-ViLT/data/f30k_ref/native/clip" 
        if debug == False:
            ans = input("Are you sure you want to modify tangram games? (y/n)")
            if ans != "y":
                os._exit(0)

        # load data file
        for sp in ["train", "dev", "test"]:
            games_dict_path = "{}/{}_image_to_texts_compact.pkl".format(folder_path, sp)
            if not os.path.exists(games_dict_path):
                games_dict_path = "{}/{}_image_to_texts.pkl".format(folder_path, sp)

            game_dict = pickle.load(open(games_dict_path, "rb"))
            outpath = "{}/f30k_{}.pkl".format(folder_path, sp)
            if sp == "train":
                new_game_dict = game_dict
                pass
            else:
                new_game_dict = {}
                for k in game_dict.keys():
                    game_examples = game_dict[k]["identifiers"][:5] + [k]
                    random.shuffle(game_examples)
                    new_game_dict[k] = game_examples

            pickle.dump(new_game_dict, open(outpath, "wb"))
"""

