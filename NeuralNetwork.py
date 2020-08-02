from ParentModel import ParentModel
from sklearn.neural_network import MLPClassifier

class NeuralNetwork(ParentModel):

    def __init__(self, df):
        super().__init__(df, MLPClassifier(random_state=1, max_iter=300))

    def trainModel(self):
        super(NeuralNetwork, self).trainModel()
        super(NeuralNetwork, self).calculateMyModelScores()
        
    def performKFoldValidation(self):
        super(NeuralNetwork, self).performKFoldValidation(MLPClassifier(random_state=1, max_iter=300))

    def getModel(self):
        return self.model