# Create a a ROC curve and a precision-recall curve
# Visually show the comparison between logreg and boosted tree models
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

#hardcode paths rather than creating configbecause I'm not resuing this I just want visuals for my readme

logreg_path = "reports/tables/modeling_v1/logreg_test_predictions_v2.csv"
xgb_path = "reports/tables/modeling_v1/xgboost_test_predictions_v1.csv"

roc_output_path = "reports/figures/modeling_v1/roc_curve_model_comparison_v1.png"
pr_output_path = "reports/figures/modeling_v1/pr_curve_model_comparison_v1.png"

Path(roc_output_path).parent.mkdir(parents=True, exist_ok=True)

logreg = pd.read_csv(logreg_path)
xgb = pd.read_csv(xgb_path)

y_logreg = logreg["actual"]
p_logreg = logreg["predicted_prob_bad"]

y_xgb = xgb["actual"]
p_xgb = xgb["predicted_prob_bad"]

#roc_curve takes true values and predictions then ouputs false positive rate, true positive rate (recall), and thresholds for each point on the curve
fpr_logreg, tpr_logreg, thresholds_logreg = roc_curve(y_logreg, p_logreg)
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_xgb, p_xgb)

#area under curves
roc_auc_logreg = round(auc(fpr_logreg, tpr_logreg), 5)
roc_auc_xgb = round(auc(fpr_xgb, tpr_xgb), 5)

# x= false positive rate, y=true positive rate (recall)
plt.figure(figsize=(8, 6))
plt.plot(fpr_logreg, tpr_logreg, label="Logistic Regression AUC = " + str(roc_auc_logreg))
plt.plot(fpr_xgb, tpr_xgb, label="XGBoost AUC = " + str(roc_auc_xgb))
# x= 0 to 1 = y so this is just the x=y line for random ranking 
plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("")
plt.legend()
plt.tight_layout()
plt.savefig(roc_output_path, dpi=150)
plt.close()

# precision_recall_curve takes same inputs and outputs % actually bad of flagged, of all bad what % were caught (recall) and thresholds
precision_logreg, recall_logreg, pr_thresholds_logreg = precision_recall_curve(y_logreg, p_logreg)
precision_xgb, recall_xgb, pr_thresholds_xgb = precision_recall_curve(y_xgb, p_xgb)

#How good the model is at ranking bad loans most risky
pr_auc_logreg = round(average_precision_score(y_logreg, p_logreg), 5)
pr_auc_xgb = round(average_precision_score(y_xgb, p_xgb), 5)
bad_rate = round(y_xgb.mean(), 5)

# x= recall and y = precision
plt.figure(figsize=(8, 6))
plt.plot(recall_logreg, precision_logreg, label="Logistic Regression PR AUC = " + str(pr_auc_logreg))
plt.plot(recall_xgb, precision_xgb, label="XGBoost v1 PR AUC = " + str(pr_auc_xgb))
# badrate
plt.axhline(y=bad_rate, linestyle="--", label="Bad-rate baseline = " + str(bad_rate))
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("")
plt.legend()
plt.tight_layout()
plt.savefig(pr_output_path, dpi=150)
plt.close()

print("Wrote:")
print(" -", roc_output_path)
print(" -", pr_output_path)