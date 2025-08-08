# This file process synthdog-en-detection dataset
import os
import json
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image, ImageDraw
# 生成数据集
# load dataset
dataset_path = "/data/dataset/data"
data = load_dataset(dataset_path, split="train")

# set save_path
image_folder = "/data/dataset/synthdog-en/data"
json_folder = "/data/dataset/synthdog-en"

# mkdir
if not os.path.exists(image_folder):
    os.makedirs(image_folder)
if not os.path.exists(json_folder):
    os.makedirs(json_folder)

converted_data = []
for id, da in enumerate(tqdm(data)):
    json_data = {}
    json_data["id"] = id
    if da["image"] is not None:
        json_data["image_name"] = f"/data/dataset/synthdog-en/data/{id}.jpg"
        da["image"].save(os.path.join(image_folder, json_data["image_name"]))
    json_data["2_coord"] = da["2_coord"]
    converted_data.append(json_data)
    
with open("/data/dataset/synthdog-en/synthdog-en.json", "w") as f:
    json.dump(converted_data, f, indent=4)

print("data process done!")