from ParentModel import ParentModel
from sklearn.naive_bayes import GaussianNB

class NaiveBayes(ParentModel):

    def __init__(self, df):
        super().__init__(df, GaussianNB())

    def trainModel(self):
        super(NaiveBayes, self).trainModel()
        super(NaiveBayes, self).calculateMyModelScores()
        
    def performKFoldValidation(self):
        super(NaiveBayes, self).performKFoldValidation(GaussianNB())

    def getModel(self):
        return self.model