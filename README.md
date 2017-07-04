---
title: "Auto HA"
author: "Ameen AboDabash"
date: "July 3, 2017"
output:
  html_document: default
  pdf_document: default
  word_document: default
---



## Executive Summary ##
Auto HA, is the automatic detection of different (H)uman (A)ctivities,i.e, we will predict "which" activity was performed at a specific point in time, using data recorded of Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

we'll walk through two different models "Decision Tree & Random Forest", and we will choose the most accurate one "maximizing the accuracy and minimizing the out-of-sample error".

As we're going to prove in this document we found that, the Random forest is the most accurate model fit our data and predict activity Type "the output variable /Class" with Overall accuracy 99~%, while the decision tree model overall accuracy is 74~%, and we're going to use Random Forest Model to predict the 20 cases in the test dataset.




## Setting the Scene ##
Loading the data "[Training](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) & [Test](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)", setting the seed, & loading the required R Packages "caret , random-Forest & rpart".



Quick Exploration analysis on training dataset , we found that there're 160 vars "Columns", its a huge number, so lets look at them and figure out which one to remove to ease our modeling process, the first column is an index column so for sure will be removed, the second column refer to participator username, so will be removed as well , we're predicting the class on activity measurements no on the username,based on same concept no need for time stamps , so lets remove time/dates cols,  finally  near Zero Vars(Columns) removed:




```r
##Loading MAster training/Testing DS, the file already downloaded to  data folder, access the src code to view the scrip
masterTrainingDS <- read.csv("data\\pml-training.csv", na.strings=c("NA","#DIV/0!",""))[-c(1:7)]
masterTestingDS <- read.csv("data\\pml-testing.csv", na.strings=c("NA","#DIV/0!",""))[-c(1,7)]


# we noticed some cols without any values all zeros, so lets remove them from the master trianing ds
masterTrainingDS<-masterTrainingDS[,colSums(is.na(masterTrainingDS)) == 0]
 
dim(masterTrainingDS)
```

```
## [1] 19622    53
```


Now, let *bootstrap* our Master training dataset into training dataset (60%) and the remaining for testing our trained model.


```r
inTrain <- createDataPartition(masterTrainingDS$classe, p=0.6, list=FALSE)
myTraining <- masterTrainingDS[inTrain, ]
myTesting <- masterTrainingDS[-inTrain, ]

dim(myTraining)
```

```
## [1] 11776    53
```

```r
dim(myTesting)
```

```
## [1] 7846   53
```
 


## Decision Tree Model ##



```r
# Building Decision Tree Model :
model1 <- rpart(classe ~ ., data=myTraining, method="class")

# Predicting using Random DecisionTree Model:
prediction1 <- predict(model1, myTesting, type = "class")

# Test results on Training Testing subset data set:
confusionMatrix(prediction1, myTesting$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2046  320   26  102   43
##          B   47  821   77   29   83
##          C   47  242 1172  134  188
##          D   78  103   91  878  112
##          E   14   32    2  143 1016
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7562          
##                  95% CI : (0.7465, 0.7656)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6905          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9167   0.5408   0.8567   0.6827   0.7046
## Specificity            0.9125   0.9627   0.9057   0.9415   0.9702
## Pos Pred Value         0.8065   0.7767   0.6573   0.6957   0.8418
## Neg Pred Value         0.9650   0.8973   0.9677   0.9380   0.9358
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2608   0.1046   0.1494   0.1119   0.1295
## Detection Prevalence   0.3233   0.1347   0.2272   0.1608   0.1538
## Balanced Accuracy      0.9146   0.7518   0.8812   0.8121   0.8374
```
 


## Random Forest Model ##


```r
# Building   Random Forest model,
#thanks for R just one line and we're done!:
model2 <- randomForest(classe ~. , data=myTraining, method="class")

# Predicting using Random Forest Model:
prediction2 <- predict(model2, myTesting, type = "class")
 
# Test results on Training Testing subset data set:
confusionMatrix(prediction2, myTesting$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2229    6    0    0    0
##          B    3 1501    3    0    0
##          C    0   11 1363   12    1
##          D    0    0    2 1272    7
##          E    0    0    0    2 1434
## 
## Overall Statistics
##                                          
##                Accuracy : 0.994          
##                  95% CI : (0.992, 0.9956)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9924         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9987   0.9888   0.9963   0.9891   0.9945
## Specificity            0.9989   0.9991   0.9963   0.9986   0.9997
## Pos Pred Value         0.9973   0.9960   0.9827   0.9930   0.9986
## Neg Pred Value         0.9995   0.9973   0.9992   0.9979   0.9988
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2841   0.1913   0.1737   0.1621   0.1828
## Detection Prevalence   0.2849   0.1921   0.1768   0.1633   0.1830
## Balanced Accuracy      0.9988   0.9939   0.9963   0.9939   0.9971
```



## Results ##
So we summarized up the solution above in the executive summary and as we proved through quick modeling using R, "Thank you again for who built these amazing R packages", the first Model **"Decision Tree" predected the classe with Accuracy : 0.7402 (95% CI : (0.7304, 0.7499))**  , while the **Random Forest predicted the target var "classe" with Accuracy :0.9949 (95% CI : (0.9931, 0.9964))**.

So, by applying the trained Random forest Model to MasterTestingDataset we will get the following results:



```r
finalResults <- predict(model2, masterTestingDS, type="class")
finalResults
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

Analysis available on [git-hub]<https://github.com/aabodabash/Coursera_ML_Assignment.git>
