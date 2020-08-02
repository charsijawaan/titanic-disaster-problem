from ParentModel import ParentModel
from sklearn import tree

class DecisionTree(ParentModel):

    def __init__(self, df):
        super().__init__(df, tree.DecisionTreeClassifier())

    def trainModel(self):
        super(DecisionTree, self).trainModel()
        super(DecisionTree, self).calculateMyModelScores()
        
    def performKFoldValidation(self):
        super(DecisionTree, self).performKFoldValidation(tree.DecisionTreeClassifier())

    def getModel(self):
        return self.model
