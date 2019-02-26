library(caret)
library(tidyverse)
dataset_00_with_header <- read.csv("Documents/walletFold/Data Scientist Test/dataset_00_with_header.csv")
#find the types of variables in dataset_00_with_header
unique(lapply(dataset_00_with_header,typeof))
wallet <- dataset_00_with_header
l <- dim(wallet)[1]
train.ndx <- sample(1:l, round(l*0.70), replace=FALSE)
trainingSet<- wallet[train.ndx,]
testingSet <- wallet[-train.ndx,]
write.csv(trainingSet,"Documents/walletFold/Data Scientist Test/walletTrainingSet.csv")
write.csv(testingSet,"Documents/walletFold/Data Scientist Test/walletTestingSet.csv")
rm(dataset_00_with_header)
trainingSet <- read.csv("Documents/walletFold/Data Scientist Test/walletTrainingSet.csv")

#remove x001.  It is an index
var2keep <- setdiff(names(trainingSet),c("X","x001"))
trainingSet <- trainingSet[,var2keep]
#Remove variables that are empty or have only one value
# Found 95 variables with near zero values
findNearZ <- nearZeroVar(trainingSet)
walletCleaned <- trainingSet[,-findNearZ]
#Find Columns with NA.  We found 41 of them
write.csv(trainingSet[1:30000,],"Documents/walletFold/Data Scientist Test/wallettrainingWithNoNZ.csv")
saveRDS(trainingSet[1:30000,],"Documents/walletFold/Data Scientist Test/wallettrainingWithNoNZ.rds")
getNAColumns<- function(naVariable){
  tp <- TRUE %in% unique(is.na(naVariable))
  return(tp)
}
colWithNA<- colnames(walletCleaned)[ apply(walletCleaned, 2, getNAColumns) ]

countNAinColumns<- function(naVariable){
  tp <- data.frame(table(is.na(naVariable)))
  tp <- subset(tp, Var1 == TRUE)
  return(tp)
}
r <- lapply(walletCleaned[,colWithNA], countNAinColumns)
v <- vector(mode = "logical",length = 0)
for (i in 1:length(r)){
  v = c(v, r[[i]]$Freq)
}
#summary of missing value variables shows a large number of missing 
# values.
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 4253   25743   34929   34775   47152   65271 
summary(v)
#Due to the problem with most of these variables are missing 
#more than 10% of data points, we will have to remove them
# as we cannot attest the quality of what we have, and due to 
# the possibility of noise which can have a negative impact
# on a simple thing as the mean
# the only variables will be keeping from all the missing variables 
# are 4 and 31  or x005 and x272, and removing the other variables
toKeep <- which(v < 10000)
variables2Remove <- colWithNA[-toKeep]
variables2keep <- setdiff(names(walletCleaned),colWithNA)
walletCleaned <- walletCleaned[,variables2keep]
#We need to perform imputation on the 2 variables
features <- setdiff(names(walletCleaned),c("y"))
response <- c("y")
#imputeVar <-preProcess(walletCleaned[,features], method = c("knnImpute"))

library(RANN)
#walletCleaned <- predict(imputeVar ,walletCleaned)
#Next Step is to decide which variables are categorical
#We will start by identifying the variable with only 0 and 1
#Find # of levels for each variable
getNumberOfLevels <- function(variable){
  return (length(table(variable)))
}
responsePos <- which(names(walletCleaned) == "y")
r <- apply(walletCleaned[,-responsePos],2,getNumberOfLevels)
dim(walletCleaned)
#After looking at the variables with more than 2 but less than 8
# we decided that we will consider only variables with 2 levels
# as categorical
#We noticed that all the variables at hand have 0 and their 
# they have an extreme distribution (power law dist)
#for variables with 80 and over levels we will scale and center
#for all other variables we will take the log than 
#scale and center

#Looking at a plot with x043 and y we notice an outlier.
#We will remove it, as some models are senstive to outliers
# The observation is  39211
#This observations has outlier not only in x043 but also with 
#281 and x279
n <- which(walletCleaned$x043 > 4000000)
walletCleaned <- walletCleaned[-n,]

#Back to the data.
r <- apply(walletCleaned,2,getNumberOfLevels)
#there are 28 categorical variables
CategoricalVariables <- names(which( r== 2))
numericalVariabletoLog <- setdiff(names(walletCleaned),CategoricalVariables)
numericalVariabletoLog <- setdiff(numericalVariabletoLog,c("y"))
write.csv(walletCleaned,"Documents/walletFold/Data Scientist Test/walletJustCleaned.csv")



