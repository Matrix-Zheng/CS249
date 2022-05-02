from sklearn import metrics
import numpy as np

def compute_eer(target_scores, nontarget_scores):
    """Calculate EER following the same way as in Kaldi.

    Args:
        target_scores (array-like): sequence of scores where the
                                    label is the target class
        nontarget_scores (array-like): sequence of scores where the
                                    label is the non-target class
    Returns:
        eer (float): equal error rate
        threshold (float): the value where the target error rate
                           (the proportion of target_scores below
                           threshold) is equal to the non-target
                           error rate (the proportion of nontarget_scores
                           above threshold)
    """
    assert target_scores.shape[0] != 0 and nontarget_scores.shape[0] != 0
    tgt_scores = np.sort(target_scores)
    nontgt_scores = np.sort(nontarget_scores)

    target_size = tgt_scores.shape[0]
    nontarget_size = nontgt_scores.shape[0]
    
    target_position = 0
    for target_position, tgt_score in enumerate(tgt_scores[:-1]):
        nontarget_n = nontarget_size * target_position / target_size
        nontarget_position = int(nontarget_size - 1 - nontarget_n)
        if nontarget_position < 0:
            nontarget_position = 0
        if nontgt_scores[nontarget_position] < tgt_score:
            break
    threshold = tgt_scores[target_position]
    eer = target_position / target_size
    return eer, threshold


def get_metrics(prediction, label):
    """Calculate several metrics for a binary classification task.

    Args:
        prediction (array-like): sequence of probabilities
            e.g. [0.1, 0.4, 0.35, 0.8]
        labels (array-like): sequence of class labels (0 or 1)
            e.g. [0, 0, 1, 1]
    Returns:
        auc: area-under-curve
        eer: equal error rate
    """  # noqa: H405, E261
    assert prediction.shape[0] == label.shape[0], (prediction.shape, label.shape)
    fpr, tpr, thresholds = metrics.roc_curve(label, prediction, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # from scipy.optimize import brentq
    # from scipy.interpolate import interp1d
    # fnr = 1 - tpr
    # eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    eer, thres = compute_eer(
        np.array([pred for i, pred in enumerate(prediction) if label[i] == 1]),
        np.array([pred for i, pred in enumerate(prediction) if label[i] == 0]),
    )
    return auc, eer

def main(prediction, label):
    print(f'Accuracy: {metrics.accuracy_score(label, prediction):.3f}')
    print(f'Balanced Accuracy: {metrics.balanced_accuracy_score(label, prediction):.3f}')
    	
    print(f'Precision: {metrics.precision_score(label, prediction):.3f}')
    print(f'Recall: {metrics.recall_score(label, prediction):.3f}')
    print(f'F1: {metrics.f1_score(label, prediction):.3f}')

    print(f'Cross-entropy Loss: {metrics.log_loss(label, prediction):.3f}')
    auc, eer = get_metrics(prediction, label)
    print('AUC: {:.3f}'.format(auc))
    print('EER: {:.3f}'.format(eer))

if __name__ == '__main__':
    prediction = np.load('/home/zzs/CS249/results/task1/dev_frame.npy')
    label = np.load('/home/zzs/CS249/data/labels/dev_frame.npy')
    main(prediction, label)