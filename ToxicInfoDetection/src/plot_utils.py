import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

import matplotlib.pyplot as plt

import config

def threshold_vs_toxic(labels, predictions, level, output_image_file):
    ''''''
    # compute metrics
    f1_score_list = []
    recall_score_list = []
    precision_score_list = []
    thresholds = np.linspace(0, 1, 50)
    for ts in thresholds:
        pred_label = (predictions > ts).astype(np.int32).tolist()
        truth_label = [1 if(v >= config.level_label_dict[level]) else 0 for v in labels]
        f1_score_list.append(f1_score(truth_label, pred_label))
        precision_score_list.append(precision_score(truth_label, pred_label))
        recall_score_list.append(recall_score(truth_label, pred_label))

    # compute the optimal threshold
    threshold_best_index = np.argmax(np.array(f1_score_list))
    threshold_best = thresholds[threshold_best_index]
    f1_best = f1_score_list[threshold_best_index]
    precision_best = precision_score_list[threshold_best_index]
    recall_best = recall_score_list[threshold_best_index]

    # plot and save
    plt.plot(thresholds, f1_score_list, color= 'green')
    plt.plot(thresholds, precision_score_list, color= 'red')
    plt.plot(thresholds, recall_score_list, color= 'blue')
    plt.plot(threshold_best, f1_best, "xr", label="Best threshold")
    plt.xlabel("Threshold")
    plt.ylabel("%s f1/precision/recall" % level)
    plt.title("Threshold vs %s \n threshold %.4f, f1 %.4f, recall %.4f, precision %.4f" %
              (level, threshold_best, f1_best, recall_best, precision_best))
    plt.legend(('f1 score', 'precision score', 'recall score'),loc='upper center', shadow= True)
    plt.savefig(output_image_file)
