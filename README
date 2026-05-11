# DataMiningProject

Turkish news text classification project built with two approaches:
- Traditional machine learning (`TF-IDF` + `Naive Bayes`, `Logistic Regression`, `Random Forest`)
- Transformer-based fine-tuning (`BERT`)

The dataset used in this project is [`savasy/ttc4900`](https://huggingface.co/datasets/savasy/ttc4900), which contains Turkish news texts labeled into 7 categories.

## Project Structure

- `proje.py`: Traditional ML pipeline with text cleaning, TF-IDF vectorization, model training/evaluation, and interactive live prediction.
- `bert_final.py`: BERT-based fine-tuning pipeline using Hugging Face `Trainer`.
- `test_trainer/`: Output directory for BERT training artifacts.

## Features

- Loads and preprocesses Turkish news text data
- Trains and compares multiple classical ML classifiers
- Reports classification accuracy
- Includes an interactive CLI test mode for manual predictions
- Fine-tunes `dbmdz/bert-base-turkish-cased` for sequence classification

## Categories

The dataset labels represent the following categories:
- 0: Politics
- 1: World
- 2: Economy
- 3: Culture-Art
- 4: Health
- 5: Sports
- 6: Technology

## Requirements

Use Python 3.9+ (recommended) and install dependencies:

```bash
pip install pandas datasets scikit-learn transformers evaluate torch numpy
```

## How to Run

### 1) Traditional ML pipeline

```bash
python proje.py
```

What it does:
- Cleans text (lowercasing, punctuation and digit removal)
- Splits train/test sets
- Applies TF-IDF vectorization
- Trains 3 models and prints accuracy
- Starts an interactive prompt for custom sentence classification

### 2) BERT fine-tuning pipeline

```bash
python bert_final.py
```

What it does:
- Tokenizes data for BERT
- Fine-tunes `dbmdz/bert-base-turkish-cased` on a subset for faster experimentation
- Evaluates and prints accuracy metrics

## Notes

- BERT training can take time depending on CPU/GPU availability.
- The current BERT script uses a reduced subset of data and 1 epoch to run faster.
- If you want higher accuracy, increase training sample size and number of epochs.

## Future Improvements

- Add precision/recall/F1 and confusion matrix reporting
- Save and reload trained models for inference
- Add hyperparameter tuning for classical models
- Train BERT on the full dataset

## License

This project is for educational and research purposes.
