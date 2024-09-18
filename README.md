![cziTD1726629616](https://github.com/user-attachments/assets/6de9c8e8-3cbf-4d22-9fae-de24d6168878)


# Table of Contents
- [Project Overview](#ProjectOverview)
- [Data Collection](#DataCollection)
- [Data Pre-Processing](#DataPre-Processing)
- [Network Architecture](#NetworkArchitecture)
- [Hyperparameter Tuning](#HyperparameterTuning)
- [Results](#Results)

# Project Overview
The primary goal of this project is to design and implement a neural network from scratch, utilizing only **NumPy**,
and apply it to the Lunar Lander game. The objective is to train the neural network to control the lander, enabling it to successfully land at a specific target location within
the game environment.</p>

The neural network will be trained on collected game data, learning to predict the necessary output variables required for successful landing. These output variables include:

**Velocity X**: Horizontal velocity of the lander.

**Velocity Y**: Vertical velocity of the lander.

<p align="justify"> The project involves multiple stages, including the construction of the neural network architecture, training and optimization, and testing the model on the Lunar Lander environment to ensure it can achieve the desired goal. 
Through this process, the neural network will learn how to make landing decisions based on the input data, optimizing for accurate and safe landings.</p>

# Data Collection

Data was collected from gameplay sessions of the Lunar Lander game and stored in a CSV file after each game. The dataset includes about 220,000 entries with four columns:

- Input Variables: X and Y distances to the target landing zone.
- Output Variables: X and Y velocities of the lander.
  
This dataset is used to train the neural network to optimize landing accuracy.

# Data Pre-processing

- **Data Type Checking**: Verified that each column in the dataset was correctly represented, ensuring accurate data types.

- **Drop Duplicates**: Removed 354 duplicate entries to improve data quality and enhance model training diversity.

- **Missing Values**: Confirmed that there were no missing values in the dataset.

- **Data Visualization**: Used KDE plots to visualize data distribution. The data is widely distributed with a high concentration near zero. This issue was addressed by normalizing the dataset as shown in figure.

- **Data Normalization**: Normalized the data to ensure all values are within the range of 0 to 1.

- **Data Splitting**: Partitioned the dataset into training (70%), validation (15%), and testing (15%) sets.
  
![Blank diagram](https://github.com/user-attachments/assets/d7bd293f-639a-453d-a191-d27ffec1532a)

 # Network Architecture

The neural network consists of three layers: an input layer, a hidden layer, and an output layer. The sigmoid function is used as the activation function throughout the network. A graphical representation of the network architecture is provided in the figure.

The code is organized into several classes, each serving a specific function:

- **Dense Class:** Creates layers and establishes connections between neurons in consecutive layers.
  
- **Sigmoid Class:** Defines the sigmoid activation function.
  
- **Loss Class:** Calculates the error between actual and predicted values.
  
- **MSE Loss Class:** Computes the Mean Squared Error (MSE) loss.
  
- **SGD Optimizer Class:** Implements the stochastic gradient descent algorithm for optimization.
  
- **Sequential Class:** Offers a streamlined method for defining and organizing the network’s structure.


![Blank board](https://github.com/user-attachments/assets/1ebb141e-6251-4d3b-ad73-258808d00a29)


# Hyperparameter Tuning

To optimize the neural network's performance, we conducted a hyperparameter tuning procedure, focusing on three parameters: learning rate, momentum for stochastic gradient descent (SGD), and the number of neurons.

- **Learning Rate:** Tested values: 0.1, 0.01.
  
- **Momentum:** Tested values: 0.1, 0.5, 0.8.
  
- **Number of Neurons:** Tested values: 4, 8, 10.
  
<p align="justify"> Initial results indicated that the number of neurons had a significant impact on performance. Specifically, 4 and 8 neurons did not perform well, while 10 neurons improved performance. Consequently, for the subsequent tests, we focused on using 10 neurons.</p>

<p align="justify"> Training and validation results for the initial parameter combinations are summarized in Table. The most effective combination for achieving optimal game performance was a learning rate of 0.1, momentum of 0.1, and 10 neurons.</p>

| Combination (LR-M-NN) | Training Loss | Validation Loss | Testing Loss |
|:---------------------:|:-------------:|:---------------:|:------------:|
|      0.1 - 4 - 0.1    |    0.1292     |     0.1284      |    0.130     |
|      0.1 - 4 - 0.5    |    0.1301     |     0.1301      |    0.130     |
|      0.1 - 4 - 0.8    |    0.1319     |     0.1325      |    0.1320    |
|      0.1 - 8 - 0.1    |    0.1291     |     0.1283      |    0.129     |
|      0.1 - 8 - 0.5    |    0.1298     |     0.1298      |    0.1298    |
|      0.1 - 8 - 0.8    |    0.1312     |     0.1319      |    0.1309    |
|     0.1 - 10 - 0.1    |    0.1290     |     0.1283      |    0.1298    |
|     0.1 - 10 - 0.5    |    0.1300     |     0.1300      |    0.1300    |
|     0.1 - 10 - 0.8    |    0.1304     |     0.1309      |    0.1302    |
|     0.01 - 10 - 0.1   |    0.1322     |     0.1317      |    0.1330    |
|     0.01 - 10 - 0.5   |    0.1303     |     0.1303      |    0.1303    |
|     0.01 - 10 - 0.8   |    0.1294     |     0.1300      |    0.1293    |


# Results


![Linkedin](https://github.com/user-attachments/assets/fe5c6010-a7fb-4ebe-9398-a8ffcc01a4e6)

