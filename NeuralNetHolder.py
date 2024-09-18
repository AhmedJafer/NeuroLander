from pickletools import decimalnl_short
from NN_scratch import Sequential ,Dense , Sigmoid_Activation_Function ,loss_error ,SGD_optimizer
import numpy as np

class NeuralNetHolder:

    def __init__(self):
        #super().__init__()
        self.model = self.load_model()
        
    def load_model(self):
        model = Sequential()
        model.add_layer(Dense(2,10))
        model.add_layer(Sigmoid_Activation_Function(alpha=0.8))
        model.add_layer(Dense(10,2))
        model.add_layer(Sigmoid_Activation_Function(alpha=0.8))
        model.load_game(r"weights.pkl",r"biases.pkl")
        return model 
    
    def normalize(self, x):
        # replace the min max values after manually finding from data
        min_val = np.array([-1101.6881038213105, 65.5017562834255])
        max_val = np.array([1095.2128406059585, 1459.1722598355652])
        return (x - min_val) / (max_val - min_val)
        
    def denormalize(self, x):
        min_vals = np.array([-7.974499016773326, -7.999358667519028])
        max_vals = np.array([7.999999999999988, 7.998973669867501])
        return np.array(x) * (max_vals - min_vals) + min_vals
    
    def predict(self, input_row):
        vals_float = np.array([float(value) for value in input_row.split(',')])
        data = self.normalize(vals_float)
        output = self.model.predict(data)
        denorm = self.denormalize(output)
        Y_Velocity,X_Velocity = denorm.flatten()
        return [X_Velocity,Y_Velocity]
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        
       