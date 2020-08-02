from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, precision_score, accuracy_score, recall_score, f1_score
import numpy as np

class ParentModel():

    def __init__(self, df, model):
        self.df = df

        self.x = self.df.drop('Survived', axis=1)
        self.y = self.df['Survived']

        self.model = model

        # splitting data into train and test data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size = 0.2, random_state = 40)
        
        self.my_model_precision = None
        self.my_model_accuracy = None
        self.my_model_recall = None
        self.my_model_f1 = None
        self.my_model_score = None

        self.cross_val_precision = None
        self.cross_val_accuracy = None
        self.cross_val_recall = None
        self.cross_val_f1 = None
        self.cross_val_score = None

    def trainModel(self):
        self.model = self.model.fit(self.x_train, self.y_train)

    def calculateMyModelScores(self):
        pred = self.model.predict(self.x_test)
        self.my_model_precision = precision_score(self.y_test, pred, average='weighted') * 100
        self.my_model_accuracy = accuracy_score(self.y_test, pred) * 100
        self.my_model_recall = recall_score(self.y_test, pred) * 100
        self.my_model_f1 = f1_score(self.y_test, pred, average='weighted') * 100
        self.my_model_score = metrics.accuracy_score(self.y_test, pred) * 100

    def performKFoldValidation(self, model):
        # now performing 10 fold cross validation
        kf = KFold(n_splits=10, shuffle=True)

        # lists to store all kfold results
        cross_val_precision_list = []
        cross_val_accuracy_list = []
        cross_val_recall_list = []
        cross_val_f1_list = []
        cross_val_score_list = []

        # splitting all data
        for train_index, test_index in kf.split(self.df):

            # seperating input from output
            x = self.df.drop('Survived', axis=1)
            y = self.df['Survived']

            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = model.fit(x_train, y_train)

            pred = model.predict(x_test)
            
            # calculating score for current split and storing            
            cross_val_precision_list.append(precision_score(y_test, pred, average='weighted'))
            cross_val_accuracy_list.append(accuracy_score(y_test, pred))
            cross_val_recall_list.append(recall_score(y_test, pred))
            cross_val_f1_list.append(f1_score(y_test, pred, average='weighted'))
            cross_val_score_list.append(metrics.accuracy_score(y_test, pred))

        self.cross_val_precision = np.mean(cross_val_precision_list) * 100
        self.cross_val_accuracy = np.mean(cross_val_accuracy_list) * 100
        self.cross_val_recall = np.mean(cross_val_recall_list) * 100
        self.cross_val_f1 = np.mean(cross_val_f1_list) * 100
        self.cross_val_score = np.mean(cross_val_score_list) * 100

    def getMyModelScores(self):
        return [self.my_model_precision, self.my_model_accuracy, self.my_model_recall, self.my_model_f1, self.my_model_score]

    def getCrossValScores(self):
        return [self.cross_val_precision, self.cross_val_accuracy, self.cross_val_recall, self.cross_val_recall, self.cross_val_recall]