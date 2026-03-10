# imports
import pandas as pd
import numpy as np
import re
from datetime import datetime 

def basic_text_structure(df, text_col, label_col):
    # (1) base checks
    assert text_col in df.columns, f"Missing text col: {text_col}"
    assert label_col in df.columns, f"Missing label col: {label_col}"

    # (2) copy + standardise
    d = df.copy()
    d[text_col] = d[text_col].astype("string")

    # 
    n_label = d[label_col].value_counts(dropna=False)
    mean_nan = d[text_col].isna().mean()
    mean_empty_text = (d[text_col].str.len() == 0).mean()

    print("unique labels + counts:\n", n_label)
    print("\nmean_nan_values:", mean_nan)
    print("\nmean_empty_strings:", mean_empty_text)

    return 


def compile_text_metrics(df_in, verbose=False, group_col=None):
      # Helper
      _word_re = re.compile(r"\b\w+\b", flags=re.UNICODE)
      _sentence_re = re.compile(r"[.!?]+")
      _digit_re = re.compile(r"\d")
      _upper_re = re.compile(r"[A-ZÄÖÜ]")
      _punct_re = re.compile(r"[^\w\s]", flags=re.UNICODE)

      df = df_in.copy()
      # Basic length metrics
      df["n_chars"] = df[TEXT_COL].str.len().fillna(0).astype(int)
      df["n_words"] = df[TEXT_COL].apply(lambda x: len(_word_re.findall(x)) if x else 0)
      df["n_sentences"] = df[TEXT_COL].apply(lambda x: len(_sentence_re.findall(x)) if x else 0)

      # Ratios / counts
      df["n_digits"] = df[TEXT_COL].apply(lambda x: len(_digit_re.findall(x)) if x else 0)
      df["n_upper"]  = df[TEXT_COL].apply(lambda x: len(_upper_re.findall(x)) if x else 0)
      df["n_punct"]  = df[TEXT_COL].apply(lambda x: len(_punct_re.findall(x)) if x else 0)

      df["digit_ratio"] = np.where(df["n_chars"] > 0, 
                                    df["n_digits"] / df["n_chars"], 
                                    0.0)
      df["upper_ratio"] = np.where(df["n_chars"] > 0, 
                                    df["n_upper"]  / df["n_chars"], 
                                    0.0)
      df["punct_ratio"] = np.where(df["n_chars"] > 0, 
                                    df["n_punct"]  / df["n_chars"], 
                                    0.0)

      return df

def print_nlp_header(name):

    print("\n")
    print("=" * 50 + "\n")
    print(f"--- {name} --- {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ---\n")
    print("=" * 50 + "\n")

    return 

def describe_text_metrics(df_in, group_col, metrics=None):
    if metrics is None:
        metrics = ["n_chars", "n_words",
                    "n_sentences", "n_digits",
                    "n_upper", "n_punct",
                    "digit_ratio","upper_ratio",
                    "punct_ratio"]
            
    df_metrics = df_in[metrics].copy()
    print_nlp_header("EDA --- DESCRIPTION TEXT METRICS")

    print("(1) overall:\n",df_metrics.describe().round(3))
    print(f"\n{'---'*50}")

    if group_col:
        print("\n(2) per group:")
        col_no = 1
            for col in metrics:
            print(f"(2_{col_no}) '{col}':\n",
                    df_metrics.groupby(group_col)[col].describe().round(3))
            print()

            col_no += 1
    return 


def distribution_text_metrics(df_in, 
                            group_col=None, 
                            label_col=None, 
                            metrics=None):
    if metrics is None:
        metrics = ["n_chars", "n_words",
                    "n_sentences", "n_digits",
                    "n_upper", "n_punct",
                    "digit_ratio","upper_ratio",
                    "punct_ratio"]
            
    df_metrics = df_in[metrics].copy()

    print_nlp_header("EDA --- DISTRIBUTION TEXT METRICS")
    print("Overall:\n", df[metrics].describe(percentiles=[0.01,0.05,0.5,0.95,0.99]).round(3).T)
    print(f"\n{'---'*50}")
    # print("")
    if label_col:
        print("per label:\n")
        for col in metrics:
            print(f"{col}:\n", d.groupby(label_col)[col].describe(percentiles=[0.01,0.05,0.5,0.95,0.99]).round(3))
            print(f"\n{'---'*50}\n")

    if group_col:
        print("per group:")
        for col in metrics:
            print(f"{col}:\n", d.groupby(group_col)[col].describe(percentiles=[0.01,0.05,0.5,0.95,0.99]).round(3))
            print(f"\n{'---'*50}\n")

    return   


def find_extreme_text(df_in, group_col):
    df = df_in.copy()

    # Sehr kurze / sehr lange Texte
    df["is_empty"] = df["n_chars"] == 0
    df["is_tiny"] = df["n_words"] <= 3
    df["is_huge"] = df["n_words"] >= df["n_words"].quantile(0.99)

    print("Extreme reports:\n", 
        df[["is_empty","is_tiny","is_huge"]].mean())

    if len(df["is_empty"]) > 0:
        print("\n'empty' reports:\n", 
            df.loc[d["is_empty"], 
            group_col].head(10))

    if len(df["is_tiny"]) > 0:
        print("\n'tiny' reports:\n", 
            df.loc[d["is_tiny"], 
            group_col].head(10))

    if len(d["is_huge"]) > 0:
        print("\n'huge' reports:\n", 
            df.loc[d["is_huge"], 
            group_col].head(3))

    return 

def quickcheck_text_differences(df_in, label_col):

    df = df_in.copy()

    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    from sklearn.linear_model import LogisticRegression

    X = df[["n_words","n_chars","n_sentences",
            "digit_ratio","upper_ratio","punct_ratio"]].to_numpy()
    y = df[label_col].to_numpy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = []
    for tr, te in cv.split(X, y):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        # Fix: Specify pos_label for binary classification with string labels
        scores.append(f1_score(y[te], 
                                pred, 
                                average="binary" if len(np.unique(y))==2 else "macro", pos_label='escalation'))

    print("f1_mean:\t", np.mean(scores))
    print("f1_std:\t", np.std(scores))
    
    return {
        "f1_mean", np.mean(scores),
        "f1_std": np.std(scores)
            }