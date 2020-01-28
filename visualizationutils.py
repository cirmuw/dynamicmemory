import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, auc, roc_curve
from scipy import interp
import numpy as np
import statannot
import matplotlib.colors as mcolors


"""
Plots a ROC Curve between target and prediction for binary classification
"""
def plot_roc_curve(target, prediction, title='ROC Curve'):
    fig, ax = plt.subplots()

    fpr, tpr, threshold = roc_curve(target, prediction)
    roc_auc = auc(fpr, tpr)
    ax.set_title(title)
    ax.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    ax.legend(loc='lower right')
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')

    return ax
