
from mlflow.tracking.fluent import log_params
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,plot_confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, cross_val_score,train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from helper import *

import dvc.api
import mlflow
import mlflow.sklearn
import logging
import warnings

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


path = 'data/processed_data.csv'
repo = "/home/mahlet/10ac/"
version = "v1"
# return to normal tag version and print in markdown
data_url =dvc.api.get_url(
    path=path,
    repo=repo,
    rev=version
)

mlflow.set_experiment('Smart_ad')


if __name__ == '__main__':
  warnings.filterwarnings("ignore")
  np.random.seed(50)
  df = pd.read_csv('/home/mahlet/10ac/Smart_Ad_AB_test/data/processed_data.csv', index_col=0)
  #scaling data using meanmax Scaler
  scaled_data = df[['experiment', 'hour', 'day', 'platform_os','browser','awarness']]
  X=scaled_data.drop('awarness', axis=1)
  Y=scaled_data['awarness']
  mlflow,log_param('data url',data_url)
  mlflow.log_param('data_version', version)
  mlflow.log_param('input_rows', df.shape[0])
  mlflow.log_param('input_cols', df.shape[1])
  mlflow.log_param('model_type','Logistic Regression')



  cv = KFold(n_splits=10, random_state=1, shuffle=True)
  # create model
  model = LogisticRegression()
  # evaluate model
  scores = cross_val_score(model, df.drop('awarness',axis=1), df['awarness'], scoring='accuracy', cv=cv, n_jobs=-1)
  # report performance
  print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

  #Spliting data into training testing and validation 
  X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.1, random_state=1)

  model.fit(X_train, Y_train)
  predicted_views = model.predict(X_test)
  acc = accuracy_score(Y_test, predicted_views)

  plot_confusion_matrix(model,X_test,Y_test, normalize='true', cmap=plt.cm.Blues)
  mlflow.log_param('acc', acc)
  with open("logistic regression_metrics.txt", 'w') as outfile:
      outfile.write("Accuracy: " + str(acc) + "\n")

  # Ploting confusion matrix
  disp = plot_confusion_matrix(
      model, X_test, Y_test, normalize='true', cmap=plt.cm.Blues)
  plt.savefig('confusion_matrix.png')