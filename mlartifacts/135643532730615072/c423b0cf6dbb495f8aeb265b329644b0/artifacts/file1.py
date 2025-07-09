import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns

'''
Note: if we try to log a local file as an artifact, MLFlow can generate an error
To avoid the same, MLFlow suggests to set the tracking URI
'''
mlflow.set_tracking_uri('http://localhost:5000')
print(mlflow.get_tracking_uri())

#load dataset
wine = load_wine()

#separate out features and target variable
X = wine.data 
y = wine.target

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state= 42)

#define parameters for RandomForest model
max_depth = 5
n_estimators = 10

#specify experiment name. If not done, then mlflow will run experiments under Default
#Note: while setting experiment name, if the experiment name specified does not exist, it will be created
mlflow.set_experiment('exp1')

#start mlflow experiment context manager
with mlflow.start_run():
    #create randomforest classifier object 
    rf = RandomForestClassifier(max_depth= max_depth, n_estimators= n_estimators)
    
    #train model
    rf.fit(X_train,y_train)

    #generate predictions on test data
    y_pred = rf.predict(X_test)

    #measure accuracy against actual test data
    accuracy = accuracy_score(y_test, y_pred)
    #print model accuracy in console
    print(accuracy)

    #create confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize= (6,6))
    sns.heatmap(cm, annot=True, fmt = 'd', cmap = 'Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")

    #save plot locally
    plt.savefig("confusion_matrix.png")

    #log model experimentation attributes in mlflow experiment
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)
    
    #log artifacts(plots, code script)
    mlflow.log_artifact("confusion_matrix.png")
    
    #the below format logs the current code script, in this case -> file1.py
    mlflow.log_artifact(__file__)

    #set tags -> this should be in dictionary format
    mlflow.set_tags({"Author": "AJ", "Project": "Wine Classification"})

    #log model
    #mlflow.sklearn.load_model(rf, "Random-Forest-Model")

