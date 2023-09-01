# python -m src.preprocessing.tangram.calculate_tangram_games
import json
import random
import os, pickle
import pyarrow as pa
import numpy as np
from tqdm import tqdm
from IPython import embed
from typing import Dict, List


batch_filepathes = [
    "./data/tangram/heldout/heldout_controlled.json",
    "./data/tangram/dev/dev_controlled.json",
    "./data/tangram/val/val_controlled.json"
]

eval_games = {}
num_distractors = 9
num_games_per_annotation = 10
for bf in batch_filepathes:
    print(bf)
    data = json.load(open(bf, "r"))
    for page in data.keys():
        for k in data[page].keys():
            data_list = data[page][k]
            for cl in data_list:
                identifier = cl["target"][0]
                game_lists = np.array(cl["distractors"])[:, :num_distractors, 0]
                assert(identifier not in eval_games)
                eval_games[identifier] = game_lists

def caculate_games(arrow_path: str, train: bool = False, debug: bool = False) -> Dict:
    games: Dict[str, List] = {} # identifier to List of identifiers

    if os.path.isfile(arrow_path):
        tables =  pa.ipc.RecordBatchFileReader(
            pa.memory_map(arrow_path, "r")
        ).read_all()

    if debug:
        if train:
            pass
        else:
            identifiers = list(tables["identifier"].to_numpy())
            for id in identifiers:
                distractors = set(identifiers)
                distractors.remove(id)
                distractors = list(distractors)
                games.setdefault(id, [])
                for _ in range(num_games_per_annotation):
                    eval_game_ids = random.sample(distractors, num_distractors)
                    eval_game_ids.append(id)  # adding a ground-truth tangram id
                    random.shuffle(eval_game_ids)  # shuggle image order
                    games[id].append(eval_game_ids)
    else:
        if train:
            pass
        else:
            identifiers = list(tables["identifier"].to_numpy())
            image_pathes = list(tables["image_path"].to_numpy())
            base_names = np.array([os.path.basename(p).replace(".png", "") for p in image_pathes])
            num_missing_ids = 0
            for i, id in enumerate(identifiers):
                base_name = base_names[i]
                exist = (base_name in eval_games)
                num_missing_ids += int(not exist)
                if exist:
                    games[id] = []
                    for eval_game in eval_games[base_name]:
                        eval_game_ids = []
                        for ctx in eval_game:
                            ctx_id = np.argwhere(base_names == ctx)[0][0]
                            eval_game_ids.append(identifiers[ctx_id])
                        eval_game_ids.append(id) # adding a ground-truth tangram
                        random.shuffle(eval_game_ids)  # shuggle image order
                        games[id].append(eval_game_ids)
                        
            print("{} / {} examples missing.".format(num_missing_ids, len(identifiers)))

    return games


if __name__ == "__main__":
    folder_path = "data/tangram/single_batch" 

    # load data file
    for debug in (True, False):
        folder_path = "data/tangram/debug" if debug else "data/tangram/native" 
        if debug == False:
            ans = input("Are you sure you want to modify tangram games? (y/n)")
            if ans != "y":
                os._exit(0)

        # load data file
        all_arrows = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if ".arrow" in f]

        for a in all_arrows:
            # calculate games
            split = os.path.basename(a).replace(".arrow", "")
            games = caculate_games(a, ("train" in split) or ("debug" in split), debug)

            # save a game mapper
            if len(games) != 0:
                game_path = os.path.join(folder_path, split + ".pkl")
                pickle.dump(games, open(game_path, "wb"))    
