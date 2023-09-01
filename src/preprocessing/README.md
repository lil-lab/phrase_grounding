# Data Preparation

## Step 1: Download Required Images

### 1.1 Touchdown SDR

- **Annotations:** Download the annotated Touchdown SDR JSONs by [clicking here](https://drive.google.com/file/d/16hNYZndOPB8W_B0b_8Yjg9_VohUREKBI/view?usp=sharing).

- **Image Sets:** For downloading panoramas and the associated streetview graph, follow the instructions offered by [Chen et al.](https://github.com/lil-lab/touchdown).

- **Image Processing:** Process the images to a perspective format and crop the upper and lower regions by executing the commands below:
```bash
# Ensure you're in the repository's root directory
cd ../..

# Run the processing script
python -m src.preprocessing.touchdown_sdr.preprocess_images --out_alias touchdown_images --json_dir JSON_DIRPATH--image_dir IMAGE_DIRPATH --graph_dir GRAPH_DIRPATH
```

### 1.2 Flickr30k Entities

- To get the Flickr30k Entities images, please consult [this repository](https://github.com/ashkamath/mdetr/blob/main/.github/flickr.md).

### 1.3 Kilogram

- Fetch the Kilogram dataset directly from [this source](https://drive.google.com/file/d/1piAZncFFsqakwFtqvC0oXwuxj5JCWWM3/view?usp=sharing).

## Step 2: Obtain Annotations

| Dataset            | Link to Download Annotations                                           |
|--------------------|-----------------------------------------------------------------------|
| Touchdown SDR      | [Access here](https://drive.google.com/file/d/1YbVwkXlLt63-QIYv7SQIeA1qN1YaE9pQ/view?usp=sharing)|
| Flickr30k Entities | [Access here](https://drive.google.com/file/d/16RNWSl5lgp8achBlR0JkikqIOzxnj0Sf/view?usp=sharing)|
| Kilogram           | [Access here](https://drive.google.com/file/d/1-2ApV7nBE8Cu4AMj8o0wLrwzX1m13Wk6/view?usp=sharing) |



## Step 3: Preprocess and Binarize Data

```bash
# Ensure you're in the repository's root directory
pwd

# For Touchdown SDR
python -m src.preprocessing.write_coco_format_datasets --mode touchdown_sdr --image_dir IMAGE_DIRPATH --annotation_dir ANNOTATION_DIRPATH

# For Flickr30k Entities
python -m src.preprocessing.write_coco_format_datasets --mode flickr --image_dir IMAGE_DIRPATH --annotation_dir ANNOTATION_DIRPATH

# For Kilogram
python -m src.preprocessing.write_coco_format_datasets --mode tangram --image_dir IMAGE_DIRPATH --annotation_dir ANNOTATION_DIRPATH
```

## Scripts and Directories
- `write_coco_format_datasets.py`: Converts datasets into COCO format arrow data files.
- `f30k/`: Contains scripts for Flickr30k Entities data preprocessing.
- `tangram/`: Scripts for Tangram data preprocessing.
- `touchdown_sdr/`: Scripts for Touchdown SDR data preprocessing.
