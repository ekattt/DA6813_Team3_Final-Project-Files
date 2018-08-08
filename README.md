# DA6813_Team3_Final-Project-Files

---
title: "Project_Final"
author: "Ekaterina Titova"
date: "7/5/2018"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

```{r}
library(dplyr)
library(sqldf)
library(psych)
library(MASS)
library(Amelia)
library(mlbench)
library(MASS)
library(ISLR)
library(tree)
library(InformationValue)
library(randomForest)
library(mboost)
library(lattice)
library(ggplot2)
library(caret)

lastword <- read.csv("/Users/ekaterinatitova/lastwords3.csv", stringsAsFactors = TRUE)

glimpse(lastword)
#let's assess if there's any pattern in missing data

missmap(lastword, col=c("red","blue"))

#according to output only 3% is missing, we are not going to remove the missing variables at this stage. We will let na.action
#command handle this within train function

#further prep the dataset by removing such columns as TCDJ number,
#first name and last name because they are providing no value to the model.
#We are also going to use a recoded version of LastStatement as we are focusing on presence vs.absense 
of words of regeret in final statement. We are going to use "1" for presence of the statement and "0" for its absence.

lastw_new <- lastword[,1:10]
lastw_new$natCounty <- as.factor(ifelse(lastw_new$natCounty == 0,0,1))

lastw_new$prevCrime <- as.factor(ifelse(lastw_new$prevCrime == 0,0,1))
lastw_new$coDefs <- as.factor(ifelse(lastw_new$coDefs == 0,0,1))



lastw_new$Sorry <- as.factor(lastw_new$Sorry)

summary(lastw_new)

```



```{r}


#Let's first split the data in training and testing samples.Since we are having a small size with factor based predictors, we are going to use the LOOCV method for trainControl function. The LOOCV is known to be a useful cross validation method when it comes to splitting categorical datasets with a relatively small size.

Sorry_sample <- createDataPartition(lastw_new$Sorry, p=3/4,list=FALSE,times=3)

Sorry_train <- lastw_new[Sorry_sample,]

Sorry_test <- lastw_new[-Sorry_sample,]

train_control <- trainControl(method="LOOCV")

set.seed(125)

logisticReg <- train(Sorry ~ .,
          data = Sorry_train,
          method = "glm",
          trControl = train_control,
          na.action = na.omit)
logisticReg$finalModel
summary(logisticReg)

#Let's explore interactions terms available for the model:

full.model <- glm(Sorry~., family = binomial, data = na.omit(lastw_new))
step.model <- stepAIC(full.model,scope=. ~ .^2, direction = "both", trace=FALSE)
summary(step.model)

#Final Model with Interaction Terms added:
#  Age + Race + edu + natCounty + prevCrime + coDefs + numVics + prevCrime:coDefs + prevCrime:numVics + coDefs:numVics + edu:natCounty + Age:natCounty + natCounty:coDefs

```

```{r}
#Boosted Generalized Linear Model

library(arm)

logisticBoost <- train(Sorry ~ .,
          data = Sorry_train,
          method = "glmboost",
          trControl = train_control,
          na.action = na.omit)
summary(logisticBoost)
print(logisticBoost)

#Selection frequencies:
RaceWhite    RaceOther RaceHispanic coDefs1 natCounty1   ageRec  Age 
0.1866      0.1200          0.1066   0.1066   0.10       0.093   0.086 
  edu      GenderVicM   numVics  (Intercept)      prevCrime1 
  0.0600   0.05333333   0.04666667   0.02000000   0.02000000

```



```{r}
#Let's try PlS model
library(AppliedPredictiveModeling)
library(stats)

set.seed(123)
plsModel <- train(Sorry ~ .,
                  data=Sorry_train,
                  method = "pls",
                  tuneLength = 10,
                  trControl = train_control,
                  na.action = na.omit)  

pls_table <- plsModel$results; pls_table

plsModel$bestTune

```



```{r}
library(caret)
library(e1071)
set.seed(1056)

svmFit <- train(Sorry ~., 
          data = Sorry_train,
          method = "svmRadial",
          tuneLength = 5,
          trControl = train_control,
          na.action = na.omit)
summary(svmFit)
print(svmFit)

                 

#let's visualize the SVM model

plot(svmFit, scales = list(x = list(log = 2)))

# based on the output, the optimal accuracy of 72% given the penalty can be achived at the cost=4 
# so far this model looks like a very promising selection

```


```{r}
#Multivariate Adaptive Regression Spline model
library(earth)
set.seed(100)

marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:15)


marsTuned <- train(Sorry ~ .,
    data= Sorry_train, 
    method = "earth",
    tuneGrid = marsGrid,
    trControl = train_control,
    na.action = na.omit)
    
marsTuned$results
#Let's use the varImp function and explore the importance variables
varImp(marsTuned)


#As we may see, the top 5 important variables are Race White, Age when received and native county
#RaceWhite     100.00
#ageRec         87.53
#natCounty1     31.81
#Age            21.72
#edu            20.08



```


```{r}

knnFit <- train(Sorry~., method="knn", data=Sorry_train, tuneLength=10, trControl=train_control,na.action = na.omit)
knnFit$results

#   k  Accuracy        Kappa
#1   5 0.6582278  0.159562204
#2   7 0.6399437  0.082367888
#3   9 0.6497890  0.080923235
#4  11 0.6540084  0.083705287
#5  13 0.6624473  0.096626644
#6  15 0.6526020  0.059115676
#7  17 0.6399437  0.020914020
#8  19 0.6399437  0.007622101
#9  21 0.6329114 -0.035662263
#10 23 0.6441632 -0.020942943

# as we can see the optimal level of accuracy is achieved at k=13
```

```{r}
ldaFit <- train(Sorry~., data=Sorry_train, method="lda", tuneLength=10, trControl=train_control, na.action=na.omit)
ldaFit$results
varImp(ldaFit)
#  parameter  Accuracy      Kappa
#1      none 0.6821378 0.08672077

#          Importance
#ageRec       100.000
#Race          90.243
#Age           66.968
#prevCrime     45.437
#coDefs        42.060
#edu           37.058
#natCounty     23.700
#GenderVic      7.381
#numVics        0.000
```

```{r}
library(mgcv)
library(nlme)
gamFit <- train(Sorry~., data=Sorry_train, method="gam", tuneLength=5, trControl=train_control, na.action=na.omit)
gamFit$results

#  select method  Accuracy     Kappa
#1  FALSE GCV.Cp 0.6821378 0.1402568
#2   TRUE GCV.Cp 0.6807314 0.1353006

#             Overall
#RaceWhite    100.000
#ageRec        81.370
#natCounty1    53.476
#coDefs1       30.845
#RaceHispanic  17.796
#Age            8.713
#numVics        3.862
#edu            3.826
#GenderVicM     2.669
#prevCrime1     0.000
```




