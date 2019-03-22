# cebd1160_project_template
Instructions and template for final projects.

| Name | Date |
|:-------|:---------------|
|Van Quang Nguyen | Completion date|

-----

### Resources
Your repository should include the following:

- Python script for your analysis: pscript.py
- Results figure/saved file: 
- Dockerfile for your experiment: Dockerfile
- runtime-instructions in a file named RUNME.md

-----

## Research Question

Among the features of the breast cancer data set, which one are more 
correlated to each other and based on this data set, can we  
predict if patients with provided information is malignant or benign.

### Abstract

Based on the UCI ML Breast Cancer Wisconsin (Diagnostic) datasets and 
visualization tools of Seaborn, we study the correlation among features 
to have the first insight on the data. From the visualization, we decide 
to use the logistic regression as a model to predicting future patients. 
We provide a python script to return a prediction once users provide 
enough information. We packed this code into a Docker container such 
that it could be used by others easily.


### Introduction

This project aims to use the Breast Cancer Wisconsin (Diagnostic) 
dataset with 569 samples. Each sample is assigned to an unique 
identification (ID) and belongs to one of two classes: Malignant (M) 
or Benign (B). One sample consists of 30 numerical-valued 
features including: radius, texture, perimeter, area, smoothness, 
compactness, concavity, concave points, symmetry, and fractal 
dimension, all are sampled with mean, standard error, and largest. 

As described on https://scikit-learn.org/stable/datasets/index.html, 
Features are computed from a digitized image of a fine needle aspirate
(FNA) of a breast mass. They describe characteristics of the cell 
nuclei present in the image


Brief (no more than 1-2 paragraph) description about the dataset. Can copy from elsewhere, but cite the source (i.e. at least link, and explicitly say if it's copied from elsewhere).

### Methods

To have the first insight on the correlation among features, we used 
seaborn and matplotlib libraries, designed for Python language, to 
plot the correlation matrix. Some pairplot will then be used to 
visualize the relation and finally, from these figures, we decided 
to use the logistic regression model in scikit-learn to build a 
classifier from which users can use to predict their class with 
their information. 

The reason to choose this regressor are because of its simplicity and 
effectiveness. Besides, from the visualization, we did not observe a 
linear relation and hence, although much simpler, linear classification 
was not be used in our project. Traditionally, logistic regression 
(even with "regression" word in its name) is a statistical model using 
logistic function (see https://en.wikipedia.org/wiki/Logistic_function) 
to deal with dependent variables with two possible outcomes which fits 
perfectly our data set (two classes). It is a modification of the 
linear classifier and has been used effectively for long-term 
(see https://amstat.tandfonline.com/doi/abs/10.1080/01621459.1975.10480319#.XJU3YC0ZM0o). 

 

Brief (no more than 1-2 paragraph) description about how you decided to approach solving it. Include:

- pseudocode for this method (either created by you or cited from somewhere else)
- why you chose this method

### Results

Brief (2 paragraph) description about your results. Include:

- At least 1 figure
- At least 1 "value" that summarizes either your data or the "performance" of your method
- A short explanation of both of the above

### Discussion
Brief (no more than 1-2 paragraph) description about what you did. Include:

- interpretation of whether your method "solved" the problem
- suggested next step that could make it better.

### References
All of the links

-------
