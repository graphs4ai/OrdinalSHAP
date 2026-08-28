import subprocess

MODELS = ["deepseek", "gemma", "mistral"]
DATASETS = ["sst-5", "yelp-20", "yelp-50"]
PAYOFFS = ["ordinal", "tf-idf", "minilm", "bm25", "google-embeddings-2", "bge", "gte", "e5"]


def run(cmd):
  print(f"$ {cmd}")
  subprocess.run(cmd, shell=True, check=True)


# E1 — ordinality preservation (Table S1)
run("python3 e1.py")

# Compute Shapley values (delete masker) for all model x payoff x dataset combinations
for model in MODELS:
  for payoff in PAYOFFS:
    for dataset in DATASETS:
      run(f"python3 compute_shapley.py --model_name {model} --masker_name delete --payoff_name {payoff} --dataset_name {dataset}")

# E2 — attribution disagreement across payoffs
for dataset in DATASETS:
  for model in MODELS:
    run(f"python3 e2_heatmap.py --model_name {model} --dataset_name {dataset}")

for dataset in DATASETS:
  for model in MODELS:
    run(f"python3 e2_analysis.py --model_name {model} --dataset_name {dataset}")

run("python3 e2_aggregate.py")

# E3 — fidelity to model behavior
for dataset in DATASETS:
  for model in MODELS:
    run(f"python3 e3.py --model_name {model} --dataset_name {dataset}")

run("python3 e3_aggregate.py")

# Compute Shapley values (replacement masker, ordinal payoff only) for E4
for model in MODELS:
  for dataset in DATASETS:
    run(f"python3 compute_shapley.py --model_name {model} --masker_name replacement --payoff_name ordinal --dataset_name {dataset}")

# E4 — robustness across masking strategies
for dataset in DATASETS:
  for model in MODELS:
    run(f"python3 e4.py --model_name {model} --dataset_name {dataset}")

run("python3 e4_aggregate.py")

# E5 - Case Study
run("python3 e5.py")