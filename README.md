# Implementation-of-SVM-For-Spam-Mail-Detection
### Date:
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1. Start the Program.

Step 2. Import the necessary packages.

Step 3. Read the given csv file and display the few contents of the data.

Step 4. Assign the features for x and y respectively.

Step 5. Split the x and y sets into train and test sets.

Step 6. Convert the Alphabetical data to numeric using CountVectorizer.

Step 7. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.

Step 8. Find the accuracy of the model.

Step 9. End the Program.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: R suraj pandian
RegisterNumber:  212223080040
*/

import pandas as pd
data=pd.read_csv("Exp_11_spam.csv",encoding='windows-1252')

data.head()

dat.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
#CountVectorizer is a method to convert text to numerical data. The text is transformed to a sparse matrix
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
### Head Function:
![1001](https://github.com/user-attachments/assets/06c9028e-91a4-4276-a910-31adf421c47e)


### Data Information:
![1002](https://github.com/user-attachments/assets/8afa097e-71f5-4975-b9eb-c15960f5b983)

### Y-Predict
![1003](https://github.com/user-attachments/assets/07b8b82b-aeb9-4a4c-a71b-959df2bc69b3)

### Accuracy:
![1004](https://github.com/user-attachments/assets/bebfeb03-4185-4135-8ed8-b86370050662)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
