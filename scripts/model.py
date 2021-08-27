from sklearn import model_selection
from scripts.helper import data_loader
from sklearn.model_selection import KFold, cross_val_score,train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score,plot_confusion_matrix
#import sys
#sys.path.insert(0, '/home/mahlet/10ac/Smart_Ad_AB_test/')

class Modeling:


    def __init__(self, filename):
        self.df = data_loader(filename)

    def data_spliter (df):

        selected_data = df[['experiment', 'hour', 'day', 'platform_os','browser','awarness']]
        X=selected_data.drop('awarness', axis=1)
        Y=selected_data['awarness']
        return X, Y

    def train (self):
         #Spliting data into training testing and validation 
         #X_train,X_test,Y_train,Y_test=train_test_split(self.X,self.Y, test_size=0.1, random_state=1)

         #model defination 
         cv = KFold(n_splits=10, random_state=1, shuffle=True)
         #create model
         model = LogisticRegression()
         # evaluate model
         scores = cross_val_score(model, self.df.drop('awarness',axis=1), self.df['awarness'], scoring='accuracy', cv=cv, n_jobs=-1)
         # report performance
         print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
   
    def train_mlflow(self):
         X,Y= Modeling.data_spliter(self.df)

         mlflow.log_param('input_rows', self.df.shape[0])
         mlflow.log_param('input_cols', self.df.shape[1])
         mlflow.log_param('model_type','Logistic Regression')

         X_train,X_test,Y_train,Y_test=train_test_split(X, Y, test_size=0.1, random_state=1)
         # create model
         model = LogisticRegression()

         model.fit(X_train, Y_train)
         predicted_views = model.predict(X_test)
         acc = accuracy_score(Y_test, predicted_views)
         #print(acc)
        
         plot_confusion_matrix(model,X_test,Y_test, normalize='true', cmap=plt.cm.Blues)
         
         mlflow.log_param('acc', acc)
         with open("logistic regression_metrics.txt", 'w') as outfile:
              outfile.write("Accuracy: " + str(acc) + "\n")
         # Ploting confusion matrix
         disp = plot_confusion_matrix(
         model, X_test, Y_test, normalize='true', cmap=plt.cm.Blues)
         plt.savefig('confusion_matrix.png')
         mlflow.log_artifact('confusion_matrix.png')
         




if __name__ =="__main__":
    model_obj= Modeling("data/processed_data.csv")
    model_obj.train_mlflow()

