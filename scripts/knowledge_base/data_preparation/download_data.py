####################################################
#
# Dataset source: https://huggingface.co/datasets/FreedomIntelligence/Disease_Database
#
####################################################

from datasets import load_dataset
import pandas as pd

# Load the dataset and saved to parquet and csv files
ds = load_dataset("FreedomIntelligence/Disease_Database", "en")
df = pd.DataFrame(ds['train'])

df.to_parquet("disease_database.parquet", index=False)
df.to_csv("disease_database.csv", index=False)
