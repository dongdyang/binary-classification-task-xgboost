# binary-classification-task-for-Kaggle-sample



(1) Running environment:
	Pandas, sklearn, Python2



(2) File descriptions:
preprcessUtil.py: is utils file for preprocess.py

preprocess.py: is the data preprocess file 
		(including to process missing data, vectorize natural language,
				error data, categorize label for object columns)

core.py: is the file to process data and cross-validation
	(including many ML learning methods)

xgboostClassifier.py: is to build a classifier for core.py

data.csv: is the original data file




(3) Running:
	"python core.py" 


(4) The best result so far:
under 3-fold cross validation for current code: 
	Precision(xgboost) = 86%
	Recall(xgboost) = 84%
	F1(xgboost) = 85%



(5) Plus,
still writing the code, this is one of the versions could be run.

