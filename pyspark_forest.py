from pyspark.sql import SparkSession
from pyspark.sql import functions as functions

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

import time

# helpful for printing output
import numpy as np
import pandas as pd

# for command line arguments
import sys

# this function takes in data and returns train data and test data that is suitable for use with PySpark random forest regressor
# uses indexing and one-hot encoding for categorical columns, and splits data into train and test datasets
def prepare_data(data):
    # .show() displays a dataframe
    #print("data:")
    #data.show()
    #data.printSchema()

    #print(pd.DataFrame(data.take(5), columns = data.columns))

    # select columns to be used with the forest. RV_percent is the label column, others are the features
    data = data.select('year', 'make', 'Body_Type', 'condition', 'odometer', 'AVG_MSRP', 'Error_percent',  'Segment',  'Age_years', 'RV_percent')

    print("data after selecting columns to use with the random forest:")
    data.show()

    # index the categorical columns (make, Segment, and Body_Type) then use one-hot encoding
    indexer = StringIndexer(inputCol = 'make', outputCol = 'make_indexed')
    indexerModel = indexer.fit(data)

    indexed_data = indexerModel.transform(data)

    encoder = OneHotEncoder(inputCol='make_indexed', outputCol="make_onehot")
    encoded_data = encoder.fit(indexed_data).transform(indexed_data)


    indexer2 = StringIndexer(inputCol = 'Segment', outputCol = 'Segment_indexed')
    indexer2Model = indexer2.fit(encoded_data)
    indexed_data2 = indexer2Model.transform(encoded_data)

    encoder2 = OneHotEncoder(inputCol='Segment_indexed', outputCol="segment_onehot")
    encoded_data2 = encoder2.fit(indexed_data2).transform(indexed_data2)


    indexer3 = StringIndexer(inputCol = 'Body_Type', outputCol = 'body_type_indexed')
    indexer3Model = indexer3.fit(encoded_data2)
    indexed_data3 = indexer3Model.transform(encoded_data2)

    encoder3 = OneHotEncoder(inputCol='body_type_indexed', outputCol="body_type_onehot")
    encoded_data3 = encoder3.fit(indexed_data3).transform(indexed_data3)
    #print("data after indexing categorical columns:")
    #encoded_data3.show()

    prepared_data = encoded_data3
    featureCols = ['year', 'make_onehot', 'body_type_onehot', 'condition', 'odometer', 'AVG_MSRP', 'Error_percent',  'segment_onehot',  'Age_years']

    #combine the features into one column
    assembler = VectorAssembler(inputCols=featureCols, outputCol = "features")
    prepared_data = assembler.transform(prepared_data)
    print("data before splitting:")
    prepared_data.show()

    # splitting data into train and test data
    trainData, testData = prepared_data.randomSplit([0.7, 0.3], seed=2023)
    print("Size of training dataset: " + str(trainData.count()))
    print("Size of test dataset: " + str(testData.count()))
    return trainData, testData


# this function takes in the train dataset, test dataset, and number of trees
# fits/trains a PySpark random forest regressor model, and evalutes on the test data, returns the predictions  
def train_and_predict(trainData, testData, treeCountParameter):
    # the featuresCol is called 'features' by default, and labelCol is called 'label' by default, numtrees is 20 by default
    # initially tested using max depth of 12 and 20, 15, and 10 trees, but for final results used maxdepth of 7 and up to 200 trees for more varied runtime
    regressor = RandomForestRegressor(featuresCol = 'features', labelCol = 'RV_percent', maxDepth=7, numTrees = treeCountParameter)

    fittedModel = regressor.fit(trainData)

    # this will add three columns to the test data: rawprediction (logits), prediction, and probability
    predictions = fittedModel.transform(testData)
    return predictions

	
if __name__ == '__main__':

    # allows user to pass in number of trees as a command line argument, otherwise defaults to 50
    treeCount = 50

    if len(sys.argv) > 1:
        treeCount = int(sys.argv[1])

    start = time.time()

    spark = SparkSession.builder.appName("Pyspark-Forest").getOrCreate()

    # read in the data
    data = spark.read.csv('cleaned_data_CS532.csv', header=True, inferSchema = True).cache()
    
    trainData, testData = prepare_data(data)

    end_preprocessing = time.time()

    #---------------------------------------------------------------------------------------------------------------------

    start_training = time.time()
    
    predictions = train_and_predict(trainData, testData, treeCount)

    # shows the top 30 rows with actual retained value percentage and predicted values
    predictions.select("RV_percent", "prediction").show(30)

    # from pyspark documentation
    evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='RV_percent')

    mean_squared_error = evaluator.evaluate(predictions)
    print("RMSE: " , mean_squared_error)

    # print out time elapsed for entire program, preprocessing, and training
    end= time.time()
    print("Total time elapsed: ", end - start)
    print("Time to preprocess: ", end_preprocessing - start)
    print("Time to train, predict, and evaluate: ", end - start_training)
