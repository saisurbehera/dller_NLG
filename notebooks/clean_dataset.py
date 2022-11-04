from datasets import load_from_disk
ds= load_from_disk("filtered_split_dprc_sentences_embeddings")
import numpy as np
ds = ds.map(lambda example: {'embeddings': example["embeddings"][0]},num_proc=32)
ds.save_to_disk("filtered_split_dprc_sentences_embeddings")