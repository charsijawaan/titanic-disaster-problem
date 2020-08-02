from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import pandas as pd
import sys
from DecisionTree import DecisionTree
from RandomForest import RandomForest
from NaiveBayes import NaiveBayes
from NeuralNetwork import NeuralNetwork
from SupportVector import SupportVector

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.threadpool = QThreadPool()
        self.setGeometry(500, 500, 800, 600)
        self.setWindowTitle("Titanic Disaster")
        self.setStyleSheet("background-color: #333333;color: #ffffff")
        self.initUI()

    def initUI(self):
        # title label
        self.titleLabel = QLabel(self)
        self.titleLabel.setText("Titanic Disaster Predictor")
        self.titleLabel.move(245, 10)
        self.titleLabel.setStyleSheet("color: #ffffff;font-weight: bold;font-size: 28px")

        # select model combo box
        self.comboBox = QComboBox(self)
        self.comboBox.setGeometry(300, 80, 250, 35)
        modelList = ['Decision Tree', 'NaiveBayes', 'Neural Network', 'Random Forest', 'Support Vector']
        self.comboBox.addItems(modelList)
        self.comboBox.setStyleSheet("font-size: 15px")

        # gender label
        self.genderLabel = QLabel(self)
        self.genderLabel.setText("Gender")
        self.genderLabel.move(50, 140)
        self.genderLabel.setStyleSheet("font-size: 17px")

        # select gender combo box
        self.genderComboBox = QComboBox(self)
        self.genderComboBox.setGeometry(150, 140, 100, 28)
        gendersList = ['Male', 'Female']
        self.genderComboBox.addItems(gendersList)
        self.genderComboBox.setStyleSheet("font-size: 15px")

        # passenger class label 
        self.PClassLabel = QLabel(self)
        self.PClassLabel.setText("P Class")
        self.PClassLabel.move(50, 200)
        self.PClassLabel.setStyleSheet("font-size: 17px")

        # passenger class combo box 
        self.PCclassComboBox = QComboBox(self)
        self.PCclassComboBox.setGeometry(150, 200, 100, 28)
        PClassList = ['1', '2', '3']
        self.PCclassComboBox.addItems(PClassList)
        self.PCclassComboBox.setStyleSheet("font-size: 15px")

        # siblings label
        self.SiblingLabel = QLabel(self)
        self.SiblingLabel.setText("Siblings")
        self.SiblingLabel.move(50, 260)
        self.SiblingLabel.setStyleSheet("font-size: 17px")

        # number of siblings input field
        self.siblingsField = QLineEdit(self)
        self.siblingsField.move(150, 260)
        self.siblingsField.setStyleSheet("font-size: 15px")

        # embarked label
        self.embarkedLabel = QLabel(self)
        self.embarkedLabel.setText("Embarked")
        self.embarkedLabel.move(50, 320)
        self.embarkedLabel.setStyleSheet("font-size: 17px")

        # embarked combo box
        self.embarkedComboBox = QComboBox(self)
        self.embarkedComboBox.setGeometry(150, 320, 100, 28)
        embarkedList = ['S', 'C', 'Q']
        self.embarkedComboBox.addItems(embarkedList)
        self.embarkedComboBox.setStyleSheet("font-size: 15px")

        # train button
        self.trainBtn = QPushButton('Train and See Results', self)
        self.trainBtn.setGeometry(70, 380, 150, 40)
        self.trainBtn.setStyleSheet("font-size: 15px; background-color: #2892ec;color: #000000")
        self.trainBtn.clicked.connect(self.trainBtnHandler)

        # result scores label
        self.resultLabel = QLabel(self)
        self.resultLabel.move(300, 130)
        self.resultLabel.setStyleSheet("font-size: 13px")

        # user data prediction
        self.userDataLabel = QLabel(self)
        self.userDataLabel.move(600, 160)
        self.userDataLabel.setStyleSheet("font-size: 15px")

        # adjusting label sizes
        self.titleLabel.adjustSize()
        self.genderLabel.adjustSize()
        self.PClassLabel.adjustSize()
        self.SiblingLabel.adjustSize()
        self.embarkedLabel.adjustSize()
        self.resultLabel.adjustSize()
        self.userDataLabel.adjustSize()  

    def trainBtnHandler(self):
        # validating fields
        try:            
            numberofSiblings = int(self.siblingsField.text())

            try:
                gender = str(self.genderComboBox.currentText())
                if gender == 'Male':
                    gender = 0
                else:
                    gender = 1

                pClass = int(self.PCclassComboBox.currentText())

                embarked = str(self.embarkedComboBox.currentText())
                if embarked == 'S':
                    embarked = 0
                elif embarked == 'C':
                    embarked = 1
                else:
                    embarked = 3

                modelName = str(self.comboBox.currentText())
                
                model = self.processData(modelName, gender, pClass, numberofSiblings, embarked)
                self.modelTest(model)

                userData = [[pClass,gender,numberofSiblings,embarked]]
                prediction = model.getModel().predict(userData)

                if(prediction[0] == 0):
                    self.userDataLabel.setText('Your Passenger Drowned')
                    self.userDataLabel.setStyleSheet("font-size: 15px;color: red")
                else:
                    self.userDataLabel.setText('Your Passenger Survived')
                    self.userDataLabel.setStyleSheet("font-size: 15px;color: green")

                self.userDataLabel.adjustSize()

            except Exception as e:
                print(e)

        except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Please Enter Valid Input")
            msg.setWindowTitle("Error")
            msg.exec_()            

    def processData(self, modelName, gender, pClass, siblings, embarked):
        # loading the dataset
        df = pd.read_csv('train.csv', sep=',')

        # droping passengers id
        df = df.drop('PassengerId', axis=1)

        # changing strings to numeric values
        df["Sex"].replace({"male": 0, "female": 1}, inplace=True)
        df["Embarked"].replace({"S": 0, "C": 1, "Q": 2}, inplace=True)

        # fillin empty values
        df["Embarked"].fillna(df["Embarked"].mean(), inplace=True)

        # seperating inputs and outputs
        x = df.drop('Survived', axis=1)
        y = df['Survived']

        model = None

        if modelName == 'Decision Tree':            
            model = DecisionTree(df)
        elif modelName == 'Naive Bayes':
            model = NaiveBayes(df)
        elif modelName == 'Neural Network':
            model = NeuralNetwork(df)
        elif modelName == 'Random Forest':
            model = RandomForest(df)
        else:
            model = SupportVector(df)
        return model

    def modelTest(self, model):
        self.resultLabel.setText("")
        model.trainModel()
        model.calculateMyModelScores()
        model.performKFoldValidation()
        
        myModelScore = model.getMyModelScores()
        crossValScore = model.getCrossValScores()

        scores = '''
        My Model: \n
        Precision = {:.2f}%\n
        Accuracy = {:.2f}%\n
        Recall = {:.2f}%\n
        F1 = {:.2f}%\n
        Overall Score = {:.2f}%\n\n

        KFold Validation: \n
        Precision = {:.2f}%\n
        Accuracy = {:.2f}%\n
        Recall = {:.2f}%\n
        F1 = {:.2f}%\n
        Overall Score = {:.2f}%\n\n
        '''.format(myModelScore[0], myModelScore[1], myModelScore[2], myModelScore[3], myModelScore[4], crossValScore[0], crossValScore[1], crossValScore[2], crossValScore[3], crossValScore[4])

        self.resultLabel.setText(scores)
        self.resultLabel.adjustSize()