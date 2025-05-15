"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset, load_from_disk # pip install datasets
from tqdm import tqdm # pip install tqdm

# ------------------------------------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset HuggingFaceFW/fineweb-edu
fw = load_from_disk("/home/sichaohe/codes/GPT2-Reproduce/fineweb_edu_complete")
import os
tiktoken_cache_dir = "/home/sichaohe/codes/GPT2-Reproduce/tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count()//2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])

import sys; sys.exit(0)

from datasets import load_dataset, load_from_disk, concatenate_datasets
from datasets import Dataset
from huggingface_hub import hf_hub_download
import os
import glob
import pandas as pd

# 加载现有的 Arrow 数据集
existing_dataset_path = "edu_fineweb10B"  # 替换为你的 Arrow 数据集路径
if os.path.exists(existing_dataset_path):
    existing_dataset = load_from_disk(existing_dataset_path)
    print(f"成功加载现有数据集，包含 {len(existing_dataset)} 条记录")
else:
    print(f"找不到现有数据集: {existing_dataset_path}")
    existing_dataset = None

# 查找所有手动下载的 Parquet 文件
parquet_dir = "manual_downloads"  # 替换为你存放 Parquet 文件的目录
parquet_files = glob.glob(os.path.join(parquet_dir, "**", "*.parquet"), recursive=True)
print(f"找到 {len(parquet_files)} 个 Parquet 文件:")
for file in parquet_files:
    print(f" - {file}")

# 读取 Parquet 文件并转换为 Dataset 对象
new_datasets = []
for file in parquet_files:
    try:
        df = pd.read_parquet(file)
        dataset = Dataset.from_pandas(df)
        new_datasets.append(dataset)
        print(f"成功加载: {file}, 包含 {len(dataset)} 条记录")
    except Exception as e:
        print(f"加载 {file} 失败: {e}")

# 将所有新数据集合并
if new_datasets:
    new_combined = concatenate_datasets(new_datasets)
    print(f"新数据集合并后包含 {len(new_combined)} 条记录")
else:
    print("没有成功加载任何新数据集")
    new_combined = None

# 将新数据集与现有数据集合并
if existing_dataset is not None and new_combined is not None:
    # 检查字段是否匹配
    existing_features = set(existing_dataset.features.keys())
    new_features = set(new_combined.features.keys())

    if existing_features == new_features:
        # 字段完全匹配，可以直接合并
        final_dataset = concatenate_datasets([existing_dataset, new_combined])
        print(f"最终合并数据集包含 {len(final_dataset)} 条记录")
    else:
        # 字段不匹配，需要处理
        print(f"字段不匹配。现有数据集: {existing_features}, 新数据集: {new_features}")
        # 找出共同字段
        common_features = existing_features.intersection(new_features)
        print(f"共同字段: {common_features}")

        # 使用共同字段选择数据
        if common_features:
            existing_subset = existing_dataset.select_columns(list(common_features))
            new_subset = new_combined.select_columns(list(common_features))
            final_dataset = concatenate_datasets([existing_subset, new_subset])
            print(f"使用共同字段合并后的数据集包含 {len(final_dataset)} 条记录")
        else:
            print("没有共同字段，无法合并")
            final_dataset = None
elif existing_dataset is not None:
    final_dataset = existing_dataset
    print(f"没有新数据集，保持现有数据集 ({len(final_dataset)} 条记录)")
elif new_combined is not None:
    final_dataset = new_combined
    print(f"没有现有数据集，使用新数据集 ({len(final_dataset)} 条记录)")
else:
    final_dataset = None
    print("没有任何数据集可用")

# 保存最终数据集
if final_dataset is not None:
    output_path = "fineweb_edu_complete"
    final_dataset.save_to_disk(output_path)
    print(f"已将合并后的数据集保存到: {output_path}")

import sys;

sys.exit()

missing_files = ["010_00000.parquet", "011_00000.parquet", "012_00000.parquet", "013_00000.parquet"]
repo_id = "HuggingFaceFW/fineweb-edu"
subdir = "sample/10BT"

for file in missing_files:
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{subdir}/{file}",
            repo_type="dataset",
            cache_dir=os.path.join(os.getcwd(), "manual_downloads")
        )
        print(f"成功下载: {file} 到 {file_path}")
    except Exception as e:
        print(f"下载 {file} 失败: {e}")

import sys;

sys.exit()

dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    "sample-10BT",
    split="train[:10%]",
    verification_mode="no_checks")

current_dir = os.getcwd()

save_path = os.path.join(current_dir, "edu_fineweb10B")
dataset.save_to_disk(save_path)

print(f"Dataset saved to {save_path}")