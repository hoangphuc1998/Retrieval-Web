import argparse
import faiss
import os
import torch
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--feature_folder", type=str, help="Folder where features are stored")
parser.add_argument("--name_folder", type=str, help="Folder where name csv are stored")
parser.add_argument("--dimension", type=int, default=2048, help="Dimensionality")
parser.add_argument("--output_folder", type=str, default="./", help="Where to save index")
args = parser.parse_args()

os.makedirs(args.output_folder, exist_ok=True)
index = faiss.IndexFlatIP(args.dimension)
name_series=[]
for file in tqdm(os.listdir(args.feature_folder)):
    path = os.path.join(args.feature_folder, file)
    name_file =  os.path.join(args.name_folder, os.path.splitext(file)[0] + ".csv")
    feature = torch.load(path).numpy()
    faiss.normalize_L2(feature)
    index.add(feature)
    filenames = pd.Series(pd.read_csv(name_file, header=None, index_col=0).iloc[:,0])
    name_series.append(filenames)

image_names = pd.concat(name_series, ignore_index=True)
image_names.to_csv(f"{args.output_folder}/names.csv", header=None)
faiss.write_index(index, f"{args.output_folder}/sajem.index")
print(f"Filename len: {len(image_names)}")
print(f"Index len: {index.ntotal}")