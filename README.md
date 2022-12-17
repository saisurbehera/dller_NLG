dller
==============================

Temporal embeddings for the retreival blocks. 

Our world is dynamic and the nature of text on the web is constantly changing. In recent years, we have seen a wide-scale adoption of Large Language Models (LLMs). Most LLMs are trained with static snapshots of knowledge bases. LLMs are not only very computationally expensive but are prone to the semantic shift of existing tokens and the sub-optimal and failed understanding of new tokens. To overcome these challenges in this paper, I in- troduce the Dynamic Large Language modEl with Retrieval (DLLER). We augment the current state-of-the-art methods (Hombaiah et al.,2021) that rely on sampling methods and incremental training with weighted retrieval blocks.


### Config environment

Configure the environment with the following keys. All the development was done on a RTX 6000 machine and I can only gurantee for it to work on it. 
```
conda create -n dller python=3.7
conda activate dller
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch-lts -c nvidia 
conda install -c conda-forge nvidia-apex
git clone https://github.com/saisurbehera/dller_NLG.git
cd dller_NLG
pip -r requirments.txt
```


All the baselines based on trained Memorizing transformers , base RETRO and others.

Warning this is pretty big for the size. 
```
gsutil cp -r gs://ss6365-coms-public-nlg/ .
```


#### Baseline Memorizing Transformer 
Baseline Memorizing transformer should look like this. This whole repo was provided by google research
```
python3 transformer/ht_main.py --gin_file=transformer/configs/base_htrans.gin --workdir=/home/saisur/meliad/memory_news/  --gin_file=/home/saisur/meliad/transformer/configs/size/small_37M.gin --gin_file=transformer/configs/options/positions_t5.gin --gin_file=transformer/configs/options/seq_512.gin --gin_file=transformer/configs/options/external_memory_32k.gin --default_data_dir=./
```
#### Transformer-XL with sliding window
```
python3 transformer/ht_main.py --gin_file=transformer/configs/base_htrans.gin --workdir=/home/saisur/meliad/memory_news/  --gin_file=/home/saisur/meliad/transformer/configs/size/small_37M.gin --gin_file=transformer/configs/options/positions_t5.gin --gin_file=transformer/configs/options/seq_512.gin --default_data_dir=./
```

Although the Meliad library looks the same in the repo, the data portion has changed slightly and needs the text_closed_qa_dataset/ which should be in the public repo.

#### Baseline GPT-2 Model with LM

You can find the completed run files in the public GPT2 repo

```
python run_clm_flax.py \
    --output_dir="./" \
    --model_type="gpt2" \
    --config_name="./gpt2" \
    --tokenizer_name="./gpt2" \
    --dataset_name="text_closed_qa_datasetg" \
    --dataset_config_name="unshuffled_deduplicated_no" \
    --do_train --do_eval \
    --block_size="512" \
    --per_device_train_batch_size="64" \
    --per_device_eval_batch_size="64" \
    --learning_rate="5e-3" --warmup_steps="1000" \
    --adam_beta1="0.9" --adam_beta2="0.98" --weight_decay="0.01" \
    --overwrite_output_dir \
    --num_train_epochs="20" \
    --logging_steps="500" \
    --save_steps="2500" \
    --eval_steps="2500" 
```


#### Baseline RETRO Model 

You can find the baseline RETRO model in the RETRO folder. The code to train the model was done on an IPYNB files in the [jupyter file](models/RETRO/train.ipynb).

### Menzi Baseline Models

Most of the code, has been an improvement from Mengzi Lang code. All the retrofitting code was based on their baseline and then improved on. Please star their repo here. The retreived component is from here. I modify them to get the temporal information.

```
https://github.com/Langboat/mengzi-retrieval-lm
```



### Datasets

We use the streamingqa dataset. For the streamingqa dataset, we filter all the news and pass in text_closed_qa_dataset/ datasets with temporal information and convert it to a db. 
We do this by running the following commands. 


We convert the dataset
```
gsutil cp -r gs://ss6365-coms-public-nlg/text_closed_qa_train/ .
cd retreival-server
./process_data.sh
```

### Model

Check out the model at:

```
gs://ss6365-coms-public-nlg/retro
```

Python load the model from 
```
import sys
sys.path.insert(0, '/workspace/dller_NLG/src/') 

from models.t5_retrofitting import  RetroModel
model = RetroModel.from_pretrained('retro')
```

To run the  model, you will need to start a retreieval server.
```
cd retreival-server
python api.py --db-path ./db --config 
```





Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
