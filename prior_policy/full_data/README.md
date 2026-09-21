# Full-data Prior Flow Reference

This directory preserves the original Prior Flow trainer, which consumes every row from an expert-inversion cache. It is the full-data reference for temporal-thinning experiments.

Use [`train_full_data_prior.py`](train_full_data_prior.py) with an existing cache. The training and inference objective are described in [`../README.md`](../README.md).

The current temporal-thinning implementation is in [`../temporal_thinning/`](../temporal_thinning/).
