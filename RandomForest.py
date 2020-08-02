from ParentModel import ParentModel
from sklearn.ensemble import RandomForestClassifier

class RandomForest(ParentModel):

    def __init__(self, df):
        super().__init__(df, RandomForestClassifier(n_estimators=10))

    def trainModel(self):
        super(RandomForest, self).trainModel()
        super(RandomForest, self).calculateMyModelScores()

    def performKFoldValidation(self):
        super(RandomForest, self).performKFoldValidation(RandomForestClassifier(n_estimators=10))

    def getModel(self):
        return self.model