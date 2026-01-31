import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, 
                             auc, 
                             precision_recall_curve, 
                             average_precision_score)
# import utils.path_helper as ph

def create_roc_auc(y_test, y_proba, roc_path):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – LogReg (Group Split)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(roc_path, dpi=150)
    plt.close()

    return 


def create_pr_curve(y_test, y_proba, pr_path):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    plt.figure()
    plt.plot(recall, precision, label=f"PR (AP = {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve – LogReg (Group Split)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(pr_path, dpi=150)
    plt.close()

    return