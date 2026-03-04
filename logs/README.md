# Experiment Logs

This directory tracks all experiment logs committed to git.

## Structure

Logs are organized as: `{generator}/{lamp_num}/{retriever}/`

Example structure:
```
logs/
├── flanT5Small/
│   ├── lamp1/
│   │   ├── bm25/
│   │   │   ├── alpha_1.log
│   │   │   ├── alpha_1_normalize.log
│   │   │   ├── alpha_2.log
│   │   │   └── alpha_2_normalize.log
│   │   └── gold/
│   │       ├── alpha_8.log
│   │       └── alpha_8_normalize.log
│   └── lamp2/
│       └── ...
```

## Content

Each experiment run generates two log files:
1. **alpha_{alpha}.log** - Experiment configuration, progress updates, and performance metrics
2. **alpha_{alpha}_normalize.log** - Normalization configuration and normalized EU statistics

## Purpose

The logs directory complements `experiment_results/` by:
- Storing persistent records of all experiment runs (committed to git)
- Tracking configuration and performance metrics
- Enabling easy debugging if experiments are interrupted (partial logs are still saved)
- Providing a searchable history of all runs

Large result files (`.json`, `.tsv`) are kept in `experiment_results/` only (not in git).
