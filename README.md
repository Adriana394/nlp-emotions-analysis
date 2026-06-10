# Emotion Classification with DistilBERT and TensorFlow

Multi-class emotion classification on the Hugging Face [`emotion`](https://huggingface.co/datasets/dair-ai/emotion)
dataset using DistilBERT. The project compares two approaches — using DistilBERT
as a frozen **feature extractor** (with XGBoost / Logistic Regression on top) and
**fine-tuning** DistilBERT end-to-end.

## Dataset

The [Emotion dataset](https://huggingface.co/datasets/dair-ai/emotion) consists of
short English messages labelled with one of six basic emotions: *sadness, joy,
love, anger, fear, surprise*. Given a message, the model classifies it into one of
these emotions.

It ships with three fixed splits:

| Split      | Examples |
|------------|----------|
| Training   | 16,000   |
| Validation | 2,000    |
| Test       | 2,000    |

The dataset is imbalanced (*joy* and *sadness* dominate), which is why the
evaluation reports a **weighted F1** score alongside accuracy.

## Project structure

The original single notebook has been split into three focused notebooks that run
in order and exchange their intermediate results through the `data/` directory:

| Notebook | Purpose | Produces / consumes |
|----------|---------|---------------------|
| [`notebooks/01_EDA.ipynb`](notebooks/01_EDA.ipynb) | Exploratory data analysis: class distribution, text length. | — |
| [`notebooks/02_Preprocessing.ipynb`](notebooks/02_Preprocessing.ipynb) | Tokenization and `[CLS]` feature extraction. | writes `data/emotions_encoded`, `data/X_*.npy`, `data/y_*.npy` |
| [`notebooks/03_Modelling.ipynb`](notebooks/03_Modelling.ipynb) | Baselines, DistilBERT fine-tuning, final evaluation. | reads the artifacts above |

The generated artifacts in `data/` are git-ignored — run `02_Preprocessing.ipynb`
to regenerate them.

## Evaluation protocol

Model selection (comparing the baselines, monitoring fine-tuning) uses the
**validation** set only. The **test** set is touched exactly once, at the end of
`03_Modelling.ipynb`, to report the final numbers. This keeps the reported
performance an honest estimate of generalization rather than a leaked,
optimistic one.

## Brief DistilBERT overview

DistilBERT is a smaller, distilled version of BERT *(Bidirectional Encoder
Representations from Transformers)*. It retains most of BERT's performance while
being significantly faster and lighter. It is trained via knowledge distillation,
where the larger BERT model transfers knowledge to the smaller DistilBERT.

## Usage

1. Install the dependencies (a GPU is strongly recommended for fine-tuning — the
   notebooks run comfortably on a free Colab T4):

   ```bash
   pip install -r requirements.txt
   ```

2. Run the notebooks in order: `01_EDA` → `02_Preprocessing` → `03_Modelling`.

The model can be adapted to other text datasets by changing the dataset and the
configuration constants at the top of each notebook.

## Suggestions and improvements

Suggestions are welcome! Feel free to fork the repository and submit a pull
request with your proposed changes.

## License

This project is licensed under the MIT License.
