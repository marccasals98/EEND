from datasets import load_dataset
ds = load_dataset("/gpfs/projects/bsc88/speech/data/raw_data/CALLHOME_talkbank/callhome", "spa")
print((ds["data"]["audio"][0]["array"]))




