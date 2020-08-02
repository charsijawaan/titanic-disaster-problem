from ParentModel import ParentModel
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class SupportVector(ParentModel):

    def __init__(self, df):
        super().__init__(df, make_pipeline(StandardScaler(), SVC(gamma='auto')))

    def trainModel(self):
        super(SupportVector, self).trainModel()
        super(SupportVector, self).calculateMyModelScores()
        
    def performKFoldValidation(self):
        super(SupportVector, self).performKFoldValidation(make_pipeline(StandardScaler(), SVC(gamma='auto')))

    def getModel(self):
        return self.model