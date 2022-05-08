Introduction

Machine learning has changed the way of living. The computer science algorithms combined with data help machines learn in the way humans learn and make decisions. Our learning has been focussed on neural networks which comes under the umbrella of deep learning. The term neural network comes from neurons. Nodes, input layers, hidden layers and output layer together compose to make the Artificial Neural Networks. 
There also develops a need of building an optimized model for the dataset. A lot of architectural engineering is needed to choose the optimized model because selecting the best hyperparameters require human intervention and time.To select the best candidate architecture model, it takes a lot of manual effort, so NAS solves this problem because it works as an expert to choose the best model from the child models we get. DARTS is one of the approaches under NAS, which we have used in our project. Our project aims to classify the 11K hands dataset images into male or female(gender based recognition). The input to the model will be the hand images and by using DARTS we need to generate the most optimized model for classifying the gender. Earlier works include the building of the Convolutional Neural Network to classify the gender using the same dataset and an accuracy of 95% was recorded. Using DARTS we aim to best tune the hyperparameters so the most optimized model could be generated which might also set a higher accuracy compared to the other deep learning methods.


Neural Architecture Search
To get the best model for some dataset, hyperparameter tuning is being taken care of. There becomes a need for optimized architectures such that there is no overfitting on the training dataset. But to come up with the best model manually becomes difficult. It becomes computationally expensive to find the optimized architecture. So, for solving this problem Neural Architecture Search is used. NAS automates the process of finding the best architecture using the search strategy in the search space. There are three important terms for Neural Architecture Search:
Search Space:
To build a valid network, the network operations and how they are connected so as to construct an optimized design for the model defines the search space.   
Search strategy:
The search space is explored in the search strategy to generate the child models which generate high performance.
Performance Estimation
As the name suggests, this measure helps to check or predict the performance of the child models obtained.
There can be multiple architectures possible from a large search space. 

For search space NAS uses cells of two types in constructing the network. 
The first one is the Normal cell in which there is the same dimension provided for input and output feature map. 
The second one is the Reduction cell in which there is a reduction of width and height of the output feature map by 2.
The NAS works on the principle to get the analysis of every child model's performance so that the optimizing algorithm can be generated using the feedback from the performance analysis. So this process to find the best child model is computationally expensive.


Dataset Modification and Pipelinging with Pytorch:

For the PyTorch execution, we want to pipeline the 11k hand's dataset with the Pytorch in light of the fact that PyTorch upholds and gives just some standard datasets like Cifar-10, Mnist, Fashion Mnist, and so forth. For this, we will convert the dataset into images and labels with which, we can proceed further in implementation.
Below is the snapshot of the source code to perform the required operation:
<center><img src="./Images/1.jpg" width="480px"></center>
