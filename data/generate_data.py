from datasets import load_dataset
import pandas as pd

dataset = load_dataset("Yelp/yelp_review_full")
df = dataset["train"].to_pandas()
df["len"] = df["text"].str.split().str.len()
final_df_20 = df[df["len"] <= 20].groupby("label", group_keys=False).sample(n=10, random_state=42).reset_index(drop=True)
final_df_50 = df[(df["len"] >= 50) & (df["len"] < 100)].groupby("label", group_keys=False).sample(n=5, random_state=42).reset_index(drop=True)
final_df_20.to_csv('yelp-20.csv', index=False)
final_df_50.to_csv('yelp-50.csv', index=False)

dataset = load_dataset('SetFit/sst5')
df = dataset["train"].to_pandas()
df["len"] = df["text"].str.split().str.len()
final_df_20 = df[df["len"] <= 20].groupby("label", group_keys=False).sample(n=10, random_state=42).reset_index(drop=True)
final_df_20.to_csv('sst-5.csv', index=False)