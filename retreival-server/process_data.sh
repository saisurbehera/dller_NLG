mkdir splits
python3 preprocess.py
python3 build_split_meta.py 
python3 build_shard.py
python3 build_db.py
python3 build_index.py