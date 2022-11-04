from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import torch
from datasets import load_from_disk
device = torch.device("cuda:0")

ds = load_from_disk("/workspace/data/filtered_small_embeddings")["train"]
def get_title_examples(examples):
    split_title = [ i.split("\n")[0] for i in examples["sentence_split"]]
    examples["title"] = split_title
    return examples
ds = ds.map(get_title_examples, batched=True, num_proc=32)

print("Loaded the Ds")
print(ds)

df = ds.to_pandas()

from collections import Counter
df["sentence_split_list"] = [ [i for i in i.split("\n")[1:] if len(i)>0] for i in df["non_split"]]

df_filterd = df[["date","hash","title","sentence_split_list"]]
filtered = []
for i in df_filterd.values:
    for j in i[3]:
        filtered.append([i[0],i[1],i[2],j])
len(filtered)

import pandas as pd
filtered_df= pd.DataFrame(filtered, columns=["date","hash","title","sentence"])

import matplotlib.pyplot as plt 
all_sizes = [len(i.split()) for i in filtered_df["sentence"]]
plt.hist(all_sizes, bins=100)

from datasets import Dataset 
dataset = Dataset.from_pandas(filtered_df)


from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import torch
torch.set_grad_enabled(False)
ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

ctx_encoder.cuda()
print(ctx_encoder.device)
device = torch.device("cuda")
ctx_encoder.eval()
with torch.no_grad():
    ds_with_embeddings = dataset.map(lambda example: {'embeddings': ctx_encoder(**ctx_tokenizer(example["sentence"], return_tensors="pt",padding=True ,truncation=True).to(device))["pooler_output"].detach().cpu().numpy()},num_proc=1)
    print(ds_with_embeddings)
    ds_with_embeddings.save_to_disk("/workspace/data/filtered_split_dprc_sentences_embeddings")