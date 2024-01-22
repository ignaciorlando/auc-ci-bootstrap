from scipy.stats import bootstrap

def roc_with_conf_intervals(y_test, y_pred_prob, n_bootstraps=1000):

    y_test = np.array(y_test)
    y_pred_prob = np.array(y_pred_prob)    
    
    rng_seed = 42  # control reproducibility
    rng = np.random.RandomState(rng_seed)
    
    bootstrapped_scores = []
    aucs = []
    tprs = []
    base_fpr = np.linspace(0, 1, 101)

    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred_prob), len(y_pred_prob))
        if len(np.unique(y_test[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = metrics.roc_auc_score(y_test[indices], y_pred_prob[indices])
        bootstrapped_scores.append(score)

        fpr , tpr, _ = metrics.roc_curve(y_test[indices].astype(np.float64), y_pred_prob[indices])
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
        #plt.plot(fpr, tpr, lw=1, alpha=0.6, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc), c = colors[i])
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    #mean_auc = metrics.auc(base_fpr, mean_tprs)
    mean_auc = np.mean(sorted_scores)
    std_auc = np.std(aucs)
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std
    
    return base_fpr, mean_tprs, base_fpr, tprs_lower, tprs_upper, mean_auc, std_auc, confidence_lower, confidence_upper


def _proportion_confidence_interval(r, n, z):
    """Compute confidence interval for a proportion.
    
    Follows notation described on pages 46--47 of [1]. 
    
    References
    ----------
    [1] R. G. Newcombe and D. G. Altman, Proportions and their differences, in Statisics
    with Confidence: Confidence intervals and statisctical guidelines, 2nd Ed., D. G. Altman, 
    D. Machin, T. N. Bryant and M. J. Gardner (Eds.), pp. 45-57, BMJ Books, 2000. 
    """
    
    A = 2*r + z**2
    B = z*sqrt(z**2 + 4*r*(1 - r/n))
    C = 2*(n + z**2)
    return ((A-B)/C, (A+B)/C)

def sensitivity_and_specificity_with_confidence_intervals(TP, FP, FN, TN, alpha=0.95):
    """Compute confidence intervals for sensitivity and specificity using Wilson's method. 
    
    This method does not rely on a normal approximation and results in accurate 
    confidence intervals even for small sample sizes.
    
    Parameters
    ----------
    TP : int
        Number of true positives
    FP : int 
        Number of false positives
    FN : int
        Number of false negatives
    TN : int
        Number of true negatives
    alpha : float, optional
        Desired confidence. Defaults to 0.95, which yields a 95% confidence interval. 
    
    Returns
    -------
    sensitivity_point_estimate : float
        Numerical estimate of the test sensitivity
    specificity_point_estimate : float
        Numerical estimate of the test specificity
    sensitivity_confidence_interval : Tuple (float, float)
        Lower and upper bounds on the alpha confidence interval for sensitivity
    specificity_confidence_interval
        Lower and upper bounds on the alpha confidence interval for specificity 
        
    References
    ----------
    [1] R. G. Newcombe and D. G. Altman, Proportions and their differences, in Statisics
    with Confidence: Confidence intervals and statisctical guidelines, 2nd Ed., D. G. Altman, 
    D. Machin, T. N. Bryant and M. J. Gardner (Eds.), pp. 45-57, BMJ Books, 2000. 
    [2] E. B. Wilson, Probable inference, the law of succession, and statistical inference,
    J Am Stat Assoc 22:209-12, 1927. 
    """
    
    # 
    z = -ndtri((1.0-alpha)/2)
    
    # Compute sensitivity using method described in [1]
    sensitivity_point_estimate = TP/(TP + FN)
    sensitivity_confidence_interval = _proportion_confidence_interval(TP, TP + FN, z)
    
    # Compute specificity using method described in [1]
    specificity_point_estimate = TN/(TN + FP)
    specificity_confidence_interval = _proportion_confidence_interval(TN, TN + FP, z)
    
    return sensitivity_point_estimate, specificity_point_estimate, sensitivity_confidence_interval, specificity_confidence_interval
