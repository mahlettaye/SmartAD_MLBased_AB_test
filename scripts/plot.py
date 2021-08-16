import matplotlib.pyplot as plt
import seaborn as sns


def plot_stat(z_scores,p_values,X):
  summary= pd.DataFrame()
  summary["Features"] = X.columns
  summary["z_score"] = z_scores
  summary["p_value"] = p_values
  sns.barplot(summary["Features"],summary["p_value"], data=summary)


def feature_importance_plot(model,x):

    plt.figure(figsize=(10,8))
    plt.title("Feature importance")
    plt.xlabel("Features")
    plt.ylabel("Importance")

    features=(model.feature_importances_)
    plt.bar(x.columns,features)
    plt.show()