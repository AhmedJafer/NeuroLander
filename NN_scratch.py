import numpy as np
import pickle   #pickle library used for loading weights 

#Dense class : used to create layers and responsible for establishing connections between each neuron in the current layer and every neuron in the subsequent layer
class Dense:

    def __init__(self, number_of_inputs, number_of_neurons):
      #initialize weights using Xavier weight initialization method
        self.w = np.random.randn(number_of_inputs, number_of_neurons) * np.sqrt(2/(number_of_inputs+number_of_neurons))
        #initialize bias
        self.b = np.zeros((1, number_of_neurons))

    # Forward pass through the layer 
    def Feed_Forward(self, inputs):
      #save the inputs in a variable for Backpropagation
        self.inputs = inputs
        #the output for all neuron
        self.output = (inputs @ self.w) + self.b

    # Backpropagation to update weights and biases
    def Backpropagation(self, derivative):
      #delta weight 
        self.dw = self.inputs.reshape(-1, 1) @ derivative
        #delta biases
        self.db = np.sum(derivative, axis=0, keepdims=True)
        #the derivative for the layer
        self.derivative = derivative @ self.w.T

#a class to calculate ReLU activation function 
class ReLU_Activation_Function:
    def Feed_Forward(self, inputs):
        self.inputs = inputs
        #the output for ReLU layer
        self.output = np.maximum(0, inputs)

    def Backpropagation(self, derivative):
      #Create a copy of the derivative to avoid modifying the original variable.
        self.derivative = derivative.copy()
        #take the derivative for the layer
        self.derivative[self.inputs <= 0] = 0

# a class to calculate Sigmoid activation function 
class Sigmoid_Activation_Function:
  def __init__(self,alpha):
    self.alpha = alpha
  def Feed_Forward(self, inputs):
    self.inputs = inputs
    #the output for the Sigmoid layer
    self.output = 1 / (1 + np.exp(-self.alpha*inputs))
  def Backpropagation(self, derivative):
    #take the derivative for the layer
    self.derivative = derivative * (1 - self.output) * self.output * self.alpha

#a class to calculate Linear Activation Function
class Linear_Activation_Function:

    def Feed_Forward(self, inputs):
        self.inputs = inputs
        self.output = inputs

    def Backpropagation(self, derivative):
      #Create a copy of the derivative to avoid modifying the original variable.
        self.derivative = derivative.copy()

#Stochastic Gradient Descent optimizer class used to update weights and biases
class SGD_optimizer:
    def __init__(self, momentum , LR=0.1):
        self.LR = LR
        self.momentum = momentum

    # Update weights and biases using SGD with momentum    
    def update_params(self,layer):
        if self.momentum != 0 :
            if not hasattr(layer, 'wm'):
              # initialize Weight momentum with zero
                layer.wm = np.zeros_like(layer.w)
                # initialize Bias momentum with zero
                layer.bm = np.zeros_like(layer.b)

            #final Weight update
            delta_w = ((self.momentum * layer.wm) - (self.LR * layer.dw))
            #final Bias update
            delta_b = ((self.momentum * layer.bm) - (self.LR * layer.db))

            #new weights for the layer
            layer.w = layer.w + delta_w
            #new biases for the layer
            layer.b = layer.b + delta_b

            #the new weight momentum value
            layer.wm = delta_w
            #the new biase momentum value
            layer.bm = delta_b

        else :
            #new weights for the layer
            layer.w = layer.w - (self.LR * layer.dw)
            #new biases for the layer
            layer.b = layer.b - (self.LR * layer.db)

# a class used to calculate Mean Square Error loss 
class MSE_Loss:

    def Feed_Forward(self, y_pred, y_true):
      #calculate the loss
        losses = np.mean((y_true - y_pred)**2, axis=-1)
        batches_loses = np.mean(losses)
        return batches_loses

    def Backpropagation(self, y_pred, y_true):
      #taking the derivative of the MSE loss 
        self.derivative = -2 * (y_true - y_pred) / len(y_pred[0])
        self.derivative = self.derivative / len(y_pred)

#a class used to calculate the loss 
class loss_error:
  def Feed_Forward(self,y_pred,y_true):
    loss = y_true - y_pred
    return loss

  def Backpropagation(self,y_pred,y_true):
    #taking the derivative of the  loss 
    self.derivative = -(y_true - y_pred)


#Sequential  Class:This class provides a convenient and intuitive approach to define and organize the network's structure
class Sequential:
    def __init__(self):
      # List to store layers of the model
        self.layers = []

#A function to add layer to the model
    def add_layer(self, layer):
        self.layers.append(layer)

    def Feed_Forward(self, x):
      # Perform forward pass through all layers
        for layer in self.layers:
            layer.Feed_Forward(x)
            x = layer.output

    def Backpropagation(self, derivative):
        # Perform backpropagation through all layers
        for layer in reversed(self.layers):
            if layer == self.layers[-1]:
                layer.Backpropagation(derivative)
                prev = layer.derivative
            else:
                layer.Backpropagation(prev)
                prev = layer.derivative

    def update_params(self, optimizer):
      # Update the parameters (weights and biases) using the  optimizer
        for i in range(len(self.layers)):
            if i == 0 or i%2 == 0 :
                optimizer.update_params(self.layers[i])

#A function to save best parameter after training stopped
    def load_parameters(self,weight_parameter,biases_parameters):
     # Load pre-trained weight and bias parameters into the model 
      for i in range(len(self.layers)):
        if i == 0 or i%2 == 0:
          self.layers[i].w = weight_parameter[self.layers[i]]
          self.layers[i].b = biases_parameters[self.layers[i]]

# used to load the weights and biases for game implementation
    def load_game(self,weight_path,biase_path):
    # Load weight and bias parameters from saved files and assign them to the model
      k=0
      with open(weight_path, 'rb') as f:
        weights = pickle.load(f)
      with open(biase_path, 'rb') as f:
        biases = pickle.load(f)
      for i,j in zip(weights.keys(),biases.keys()):
        self.layers[k].w = weights[i]
        self.layers[k].b = biases[j]
        k=k+2

# used to save the parameters (weights and biases)
    def save_parameters(self,weights,biases):
      with open('weights.pkl', 'wb') as file:
        pickle.dump(weights, file)
      with open('biases.pkl', 'wb') as file:
        pickle.dump(biases, file)

    def train(self, x_train, y_train,x_val,y_val, loss_function, optimizer, epochs,threshold,best_parameters=True):
      # List to store training loss 
        self.train_graph=[]
        # List to store validation loss 
        self.val_graph =[]
        #Dict to save layer and its weights
        self.best_weights ={}
        #Dict to save the best biases
        self.best_biases = {}
        #intial the loss
        self.best_loss= 10000000
        # Epoch at which the best loss was achieved
        best_epoch = 0
        #transfer the variable to numpy array
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_val = np.array(x_val)
        y_val = np.array(y_val)

         # Training loop
        for epoch in range(epochs):
            for i in range(len(x_train)):
              #forward pass through the network
                self.Feed_Forward(x_train[i])
                #output of the last layer
                train_output = self.layers[-1].output
                #calculate the loss
                loss_function.Backpropagation(train_output,y_train[i])
                #backpropagation procedure
                self.Backpropagation(loss_function.derivative)
                #Update parameters
                self.update_params(optimizer)

            # Create an instance of the Mean Squared Error loss function
            mse = MSE_Loss()
            
            # Perform forward pass on validation and training data
            self.Feed_Forward(x_val)
            validation_output = self.layers[-1].output
            self.Feed_Forward(x_train)
            train_output = self.layers[-1].output
            
            #calculate the mse loss for training data
            train_loss =mse.Feed_Forward(train_output, y_train)
            #calculate the mse loss for validation data
            valid_loss = mse.Feed_Forward(validation_output, y_val)

            #Stopping criteria
            # Check if the current validation loss is better than the previous best loss
            if round(valid_loss,4) < round(self.best_loss,4):

              self.best_loss = valid_loss
              best_epoch = epoch

              for layer in self.layers:
                # Skip layers that don't have weights and biases
                if not hasattr(layer, 'wm'):
                  continue
                 # Store the best weights and biases for each layer  
                self.best_weights[layer] = layer.w
                self.best_biases[layer] = layer.b

            # Check if the number of epochs without improvement exceeds the threshold
            if threshold < epoch - best_epoch:
              break

            #append the training loss and validation loss 
            self.train_graph.append(train_loss)
            self.val_graph.append(valid_loss)



            #mean_loss = total_loss / len(x_train)
            print(f"Epoch {epoch + 1}/{epochs}, train_Loss: {train_loss:.4f}, val_loss: {valid_loss:.4f}")

         #Save the best weights and biases   
        if best_parameters:
            self.load_parameters(self.best_weights,self.best_biases)
            self.save_parameters(self.best_weights,self.best_biases)
        else:
          weights={}
          biases={}

          for layer in self.layers:
            if not hasattr(layer, 'wm'):
              continue
            weights[layer] = layer.w
            biases[layer] = layer.b

          #save the weight and biases from last iteration
          self.save_parameters(weights,biases)

    
    #to Make predictions using the trained model.
    def predict(self, x):
        self.Feed_Forward(x)
        return self.layers[-1].output

    #to evaluate the trained model.
    def evaluation(self,x,y):
      x=np.array(x)
      y=np.array(y)
      self.Feed_Forward(x)
      output = self.layers[-1].output
      return MSE_Loss.Feed_Forward(self,output,y)



"""
"This section outlines the neural network architecture. To utilize this code for training the model,
uncomment this section and integrate it with 'Data_Preprocessing.ipynb"


model = Sequential()
model.add_layer(Dense(2,10))
model.add_layer(Sigmoid_Activation_Function(alpha=0.8))
model.add_layer(Dense(10,2))
model.add_layer(Sigmoid_Activation_Function(alpha=0.8))



model.train(x_train = X_train, y_train= y_train,x_val=X_valid,y_val=y_valid, loss_function= loss_error() , optimizer=SGD_optimizer(momentum=0.1), epochs=100,threshold=10)

"""