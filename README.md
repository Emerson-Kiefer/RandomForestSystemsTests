# Matthew Spahl, Emerson Kiefer, Kevin Martell Luya, Owen Tibby
# CS 532 Final Project

# This program reads in a dataset of used car pricing information, trains a random forest on that dataset,
# and then predicts the RV_percent (retained value percent) for each sample in the test data.


# references to learn how to use a random forest classifier in Pyspark:
#https://towardsdatascience.com/a-guide-to-exploit-random-forest-classifier-in-pyspark-46d6999cb5db
#https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.RandomForestRegressor.html
#https://www.machinelearningplus.com/pyspark/pyspark-onehot-encoding/
#https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.RegressionEvaluator.html
#https://stackoverflow.com/questions/7370801/how-do-i-measure-elapsed-time-in-python

Instructions for running code: in an Ubuntu VM with python and pip installed, install numpy and pandas.
To run, use command python3 Pyspark_forest.py

By default, the program will train/fit the random forest model using 50 trees. To change this, either change
the value of treeCount on line 35 of the program or provide a command line argument. For example, to run with
100 trees, use the command:
python3 Pyspark_forest.py 100

The program will process the data and train a random forest regressor model and make predictions, then output
the total runtime of the program, the data processing time, and the time for train/fit the random forest,
make predictions, and evaluate the model.

Also, in our analysis for our presentation, referenced spark.apache.org/docs/2.2.0/mllib-ensembles.html to 
confirm that Pyspark Random Forest is able to train trees in parallel.
