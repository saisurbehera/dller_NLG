from datasets import load_from_disk
ds = load_from_disk("../text_closed_qa_train")

text_l = ds["train"]["text"] + ds["validation"]["text"]
dates = text = ds["train"]["date"] + ds["validation"]["date"]
text_dict = [{'meta': {'pile_set_name': 'Pile-CC' },'date':date,'text': text} for text,date in zip(text_l,dates)]

with open('splits/db.jsonl', 'w') as outfile:
    for entry in JSON_file:
        json.dump(entry, outfile)
        outfile.write('\n')