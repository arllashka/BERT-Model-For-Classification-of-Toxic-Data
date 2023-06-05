# BERT Model For Classification of Toxic Data

### Introduction ###

As a newcomer to data science, and particularly the domain of transformer models, I've been intrigued by the complexities of BERT. When studying BERT on Kaggle, I noticed that the resources available primarily explain BERT's architecture or offer code snippets with limited descriptions. This project strives to bridge that gap by linking basic BERT theory with practical application using a simple model. The framework of this work is derived from an instructive tutorial found on Kaggle (https://www.kaggle.com/chumajin/pytorch-bert-beginner-s-room), and I've adjusted it to address our specific competition problem. I've also included additional visualizations to trace the model's performance over epochs and different folds.

### Objective ###
The aim is to develop a model that can predict the likelihood of each category of toxicity in any given comment.

This notebook is divided into two parts due to Kaggle's memory restrictions. Each part can function independently. In part I, I explain the basic steps for managing binary classification tasks involving multiple labels. This model operates on a single distribution of the training set and validation set, offering an initial insight into possible performance results (AUC score).

In part II, I've retained necessary code components from part I and demonstrated how we can construct five models using different choices of the training set and validation set, following the k-fold (k=5) method. The toxicity probability for each comment is then calculated as the mean value from these five models.

### Factors Affecting Simulation Time ###
For quick execution of this notebook, both parts I and II allow you to limit the training set to a certain number of rows (200 in part I and 2000 in part II). The limitation in part I is due to its instructional nature, which necessitates the inclusion of several variables and can lead to a CUDA memory crash for larger row numbers, yielding moderate results for part I (accuracy: 86%).

Part II is more streamlined, allowing an increase in the number of rows potentially up to the complete training set. However, it's capped at 2000 rows to deliver reasonably good results (94% accuracy and 0.988 AUC score) within a 22-minute simulation.

To observe improvements in accuracy and losses across different epochs, you can adjust parameters to (epochs = 5, k = 5). This notebook doesn't aim for the highest results, but rather a satisfactory level of performance.
 
