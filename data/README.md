# Data

This directory contains the script used to generate the datasets from their original sources.

## Licensing note on Yelp

The Yelp Review Full dataset is subject to its own usage terms, which may restrict redistribution. **We do not distribute `yelp-20.csv` / `yelp-50.csv` with this repository.** To reproduce them:

```bash
cd data
python3 generate_data.py
```

This downloads `Yelp/yelp_review_full` and `SetFit/sst5` from the Hugging Face Hub and regenerates all three CSVs with the exact sampling (`random_state=42`) used in the paper.