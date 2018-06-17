# Welcome to the neoEYED Coding challenge!

Here is the problem statement: 
	

We have a dataset that contains data samples coming from 4 different users. 
	
Each user is identified from the "user_id" column
	
Each samples is marked either as 1 or -1 in the "expected_result" columns. 
Where 1 is a sample coming from the user, -1 is a sample coming from a fraudster
	

We need to create an algorithm to predict, for each user, each sample, 
if the sample is coming from the user or from the fraudster (detect if should be market 1 or -1) 
	
	


Dataset information: 
	Each sample has these columns: 
	 
- user_id: the user identificator
	 
- index: a progressive unique identificator of the sample
	 
- expected_result: for that user, identifies if the sample is a fraudster or the actual user
	 
- timestamp: a time when the sample was collected (we suggest not to use it)
	 
- matrix: this contain the actual data of the sample in a matrix format
 
 


The matrix
	
Each sample contains a matrix of data that should be used to do the train/prediction. 
	
Inside each matrix there are 27 arrays of time-series ordered data sampled at a specific frequency. 
	
Each array represent a single factor like the accelerometer data, gyroscope, magnetometer...
	
Each array has the same length as each element was taken at a specific tick, defined in the first array of the matrix
	
Some of these arrays are correlated as there is a x,y,z coordianate (for example for accelerometer data)
	
	
	



Restriction and limitation:
	
It should be necessary to split the dataset for each user and test/train each user set separately. Besides, algorithm that make use of DeepLearning or AutoEncoders may need to use the entire dataset
	You are only allowed to train your classifier using samples marked as "1" in expected_result column. 

Samples marked as "-1" can only be used for testing. For this reason multiclass classifiers couldn't be used
	


Goal:
	The goal is to create an algorithm that, trained using user positve samples, is able predict both the negative (frauds) and positive (user) samples
	with a False Acceptance Ratio of 0% and a False Rejection Ratio < 15%
	
