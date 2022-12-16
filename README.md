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


All the baselines based on trained Memorizing transformers , base RETRO and others
```
gsutil cp -r gs://ss6365-coms-public-nlg/ .
```

Baseline Memorizing transformer should look like this
```
python3 transformer/ht_main.py --gin_file=transformer/configs/base_htrans.gin --workdir=/home/saisur/meliad/memory_news/  --gin_file=/home/saisur/meliad/transformer/configs/size/small_37M.gin --gin_file=transformer/configs/options/positions_t5.gin --gin_file=transformer/configs/options/seq_512.gin --gin_file=transformer/configs/options/external_memory_32k.gin --default_data_dir=./
```

T5 Model
```
python3 transformer/ht_main.py --gin_file=transformer/configs/base_htrans.gin --workdir=/home/saisur/meliad/memory_news/  --gin_file=/home/saisur/meliad/transformer/configs/size/small_37M.gin --gin_file=transformer/configs/options/positions_t5.gin --gin_file=transformer/configs/options/seq_512.gin --default_data_dir=./
```

Although the Meliad library looks the same in the repo, the data portion has changed slightly and needs the text_closed_qa_dataset/ which should be in the public repo.


### Datasets

My datasets are fairly standard. I will be combining two different datasets. These include Pile for general language modelling. For testing the models, I will be using the large Twitter dataset and StreamingQA  dataset. 

StreamingQA is a human-written and generated questions datasets which are answered from 14 years of time-stamped news articles. Both these datasets have explicit timestamps to help guide LLMs. 

### Models


One of the most surprising changes in the LLM space has been the addition of explicit memory. These auto-regressive language models are conditioned on document chunks retrieved from a
large corpus, based on local similarity with preceding tokens. One main advantage of these approaches is the relative simplicity and performance matching of models with 25x less parameters. This approach allows us to effectively skip the retraining part of LLMs. 

Recent works by Deepmind have introduced us to Retrieval-Enhanced Transformer (RETRO). Retro combines a frozen Bert
retriever, a differentiable encoder, and a chunked cross-attention mechanism to predict tokens based on an order of magnitude more data than what is typically consumed during training. 



### Novel Extensions


One of the significant drawbacks of the retrieval block is every query is given the same regard in the chunked attention block. One of the goals of the project will be to augment this  block with information relating to time. 

This approach is a natural extension of positional embeddings applied to the retrieved block. Positional embeddings combine positional information with semantic information. Time can be thought of as a positional in another latent space. 

The downstream effects of the task will be immense as the natural extension of time is quality or other ranking information. Although widely used in the Information Retrieval (IR) tasks, these have not been generalized to the NLP field. 

### Baselines

The current baselines of the dynamic language modelling include:

* Base T-5 Model
* Retrieval Aigmeneted Generation (RAG) for Knowledge Intensive tasks
* Fusion-in-Decoder (FID) for Knowledge Intensive tasks
* Retrieval-Enhanced Transformer (RETRO) 


We will be using a baseline of T5 model. T5 model is a closed model and is trained on data as a snapshot. This is expected to give us the lowest score. 

RAG model combines two different approaches which include pre-trained sequence models and non-parametric retrieval. A parametric memory is a pre-trained seq2seq model and a non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever. All the retrieved passages are processed in the encoder. 

Fusion-in-Decoder augments the  RAG model by encoding all the retrieved texts and concats the compressed representations. The main difference with RAG is the encoder trains every retrieved block separately. 



## Evaluation

We will be evaluating all the models on the StreamingQA dataset. It contains  14 years (2007–2020) of English WMT news together with their publication dates, as our knowledge corpus (approx. 11M articles). The main benefit of the task is it focuses on both temporal effects. The task is split into quarters with dates. When the lag is negative, the model knowledge is lagging behind a question date (QD). This means that the model is missing information and has not considered recent events. When the lag is positive, the previous information has been overwritten and the model has forgotten previous results. 
 
The dataset uses the same metrics as question-answering tasks. These include F1 and the exact match (EM). Although this is the language modeling task, it would be really great if the retrieval block can be evaluated separately. These would include metrics like Mean average precision and Normalized Discounted Cumulative Gain. 

## Experiments

Retrieval-based methods have experimental evidence suggesting combining vectors in the encoder or passing them separately. Although FID and DAG models are Encoder-decoder architecture, RETRO is a decoder architecture. There is not a public benchmark comparing both these benchmarks. 

A major chunk of the time will be spent on ranking the retrieval query and passing temporal information less explicitly. The model needs to understand today as 28th September not like GPT-3 March 2020. 


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
