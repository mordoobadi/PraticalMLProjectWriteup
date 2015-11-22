
require(caret)
require(rpart) 
require(rpart.plot)
require(rattle)
require(randomForest)
require(xgboost)

trainDataSet <- read.csv("pml-training.csv", na.strings = c("NA", "#DIV/0!", ""))
testDataSet  <- read.csv("pml-testing.csv", na.strings = c("NA", "#DIV/0!", ""))
summary(trainDataSet)
#Data Cleanup

numTrainingRecords <- dim(trainDataSet)[1]
feauture.names <- names(trainDataSet)
numFeatures <- length(trainDataSet)

columnsToRemove <- c()

## Remove columns that have many NA's (more than 50% of number of records)
for(iCol in 1:numFeatures) {
  if(sum(is.na(trainDataSet[, iCol])) > 0.50 * numTrainingRecords) {
    columnsToRemove <- c(columnsToRemove, iCol)
  }
}
  
trainDataSet <- trainDataSet[, -columnsToRemove]

# remove columns X and cvtd_timestamp
trainDataSet <- trainDataSet[, -c(1, 5)]

# Replace yes/no with 1/0 in the new_window column
trainDataSet$new_window <- as.character(trainDataSet$new_window)
trainDataSet[trainDataSet$new_window == "no", "new_window"] <- "0"
trainDataSet[trainDataSet$new_window == "yes", "new_window"] <- "1"
trainDataSet$new_window <- as.integer(trainDataSet$new_window)

testDataSet$new_window <- as.character(testDataSet$new_window)
testDataSet[testDataSet$new_window == "no", "new_window"] <- "0"
testDataSet[testDataSet$new_window == "yes", "new_window"] <- "1"
testDataSet$new_window <- as.integer(testDataSet$new_window)

summary(trainDataSet)

## Split the training data to training and cross validation data sets
set.seed(0)

inTrain <- createDataPartition(y=trainDataSet$classe, p=0.70, list=FALSE)

training <- trainDataSet[inTrain, ]
testing <- trainDataSet[-inTrain, ]

# Using Decision Trees
modelFitDT <- rpart(classe ~ ., data=training, method="class")

predsDT <- predict(modelFitDT, testing, type="class")

confusionMatrix(predsDT, testing$classe)


# Using Random Forests
modelFitRF <- randomForest(classe ~., data=training)

predsRF <- predict(modelFitRF, testing, type = "class")

confusionMatrix(predsRF, testing$classe)


# Using XGBOOST
training[training$classe == "A", "label"] <- 0
training[training$classe == "B", "label"] <- 1
training[training$classe == "C", "label"] <- 2
training[training$classe == "D", "label"] <- 3
training[training$classe == "E", "label"] <- 4

testing[testing$classe == "A", "label"] <- 0
testing[testing$classe == "B", "label"] <- 1
testing[testing$classe == "C", "label"] <- 2
testing[testing$classe == "D", "label"] <- 3
testing[testing$classe == "E", "label"] <- 4

dval<-xgb.DMatrix(data=data.matrix(testing[, -c(58:59)]),label=testing$label)
dtrain<-xgb.DMatrix(data=data.matrix(training[, -c(58:59)]),label=training$label)
watchlist<-list(val=dval,train=dtrain)

param <- list(  objective           = "multi:softmax", 
                booster             = "gbtree",
                silent              = 0,
                eta                 = 0.1,
                max_depth           = 10, #changed from default of 8
                num_class = 5
)

modelFitXGB <- xgb.train(param, dtrain, 100, watchlist )

predsXGB <- predict(modelFitXGB, data.matrix(testing[, -c(58:59)]))

confusionMatrix(predsXGB, testing$label)


## Combined Models
predsRFXGB <- data.frame(predsRF, predsXGB, label=testing$label)

predsRFXGB[predsRFXGB$predsRF == "A", "labelRF"] <- 0
predsRFXGB[predsRFXGB$predsRF == "B", "labelRF"] <- 1
predsRFXGB[predsRFXGB$predsRF == "C", "labelRF"] <- 2
predsRFXGB[predsRFXGB$predsRF == "D", "labelRF"] <- 3
predsRFXGB[predsRFXGB$predsRF == "E", "labelRF"] <- 4
predsRFXGB <- predsRFXGB[-1]

modelFitCombined <- train(label ~., method="gam", data=predsRFXGB)
predsCombined <- predict(modelFitCombined, predsRFXGB)
predsCombined <- round(predsCombined)

confusionMatrix(predsCombined, predsRFXGB$label)



outOfSampleError <- (1 - sum(predsRF == testing$classe)/length(predsRF))*100

outOfSampleError

predsSubmission <- predict(modelFitRF, testDataSet, type = "class")

for(iSubmit in 1:dim(testDataSet)[1]){
  fileName = paste("problem_id_", iSubmit, ".txt", sep = "")
  write.table(predsSubmission[iSubmit], file = fileName, col.names = FALSE, row.names = FALSE, quote = FALSE)
}



