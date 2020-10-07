#Data imported from the workspace
imgs<-read.csv("hw03_images.csv",header = FALSE)
lbls<-read.csv("hw03_labels.csv",header = FALSE)
vih<-read.csv("initial_V.csv",header = FALSE)
wih<-read.csv("initial_W.csv",header = FALSE)
#Image and label data divided into two as training and test
yTrain<-lbls[c(1:(nrow(lbls)/2)),]
yTest<-lbls[c((nrow(lbls)/2)+1):nrow(lbls),]
trainSet<-imgs[c(1:(nrow(imgs)/2)),]
testSet<-imgs[c((nrow(imgs)/2)+1):nrow(imgs),]
#Data are combined just for holding together
dataSet<-cbind(imgs,lbls)
#Training labels are encoded
yTrainOptimazed<-data.frame()
a<-1
while(a<=length(yTrain)){
  if(yTrain[a]==1) yTrainOptimazed<-rbind(yTrainOptimazed,c(1,0,0,0,0))
  else if(yTrain[a]==2)yTrainOptimazed<-rbind(yTrainOptimazed,c(0,1,0,0,0))
  else if(yTrain[a]==3)yTrainOptimazed<-rbind(yTrainOptimazed,c(0,0,1,0,0))
  else if(yTrain[a]==4)yTrainOptimazed<-rbind(yTrainOptimazed,c(0,0,0,1,0))
  else if(yTrain[a]==5) yTrainOptimazed<-rbind(yTrainOptimazed,c(0,0,0,0,1))
  a<-a+1
}
#Sigmoid Function
sigmoidFunc <- function(X, w) {
  return (1 / (1 + exp(-(as.matrix(X) %*% as.matrix(w)))))
}
#Softmax Function
softmaxFunc <- function(X, V) {
  Xn<-cbind(1,X)
  scores <- as.matrix(Xn) %*% as.matrix(V)
  scores <- exp(scores - matrix(apply(scores, MARGIN = 1, FUN = max), nrow = nrow(scores), ncol = ncol(scores), byrow = FALSE))
  scores <- scores / matrix(rowSums(scores), nrow(scores), ncol(scores), byrow = FALSE)
  return (scores)
}
#Constatn Values
eta<- 0.0005
epsilon<-1e-3
maxIteration<- 500
#Gradient functions are implemented as follows
gradientW <- function(X, Ytruth, Ypredicted, v, z) {
  tmp<-rowSums(as.matrix((Ytruth-Ypredicted))%*%as.matrix(t(v)))
  tmp<-tmp*z*(1 - z)
  XT<-t(X)
  return(-(as.matrix(XT)%*%as.matrix(tmp)))
}
gradientV <- function(Ytruth, Ypredicted,z) {
  zn<-cbind(1,z)
  return (-(as.matrix(t(zn))%*%as.matrix((Ytruth - Ypredicted))))
}
#Safe log function is used to get rid of NaNs
safeLog<- function(x){return (log(x + 1e-100))}
#Column of 1s is added to the training set data
trainSetN<-cbind(1,trainSet)
iteration <- 0
objectiveValuesTraining <- c()
while (iteration<maxIteration) {
  z<-sigmoidFunc(trainSetN,wih)
  yPredTrain<-softmaxFunc(z,vih)
  objectiveValuesTraining <-c(objectiveValuesTraining,-sum(yTrainOptimazed*safeLog(yPredTrain)))
  wih <- wih - eta * gradientW(trainSetN, yTrainOptimazed, yPredTrain, vih, z)
  vih <- vih - eta * gradientV(yTrainOptimazed, yPredTrain,z)
  iteration <- iteration + 1
  print(iteration)
}
#Plotting the errors vs iteration graph
plot(1:iteration, objectiveValuesTraining,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")
#Calculating the latest version of the predicted values
ypredicted <- apply(yPredTrain, MARGIN = 1, FUN = which.max)
#Confusion matrix is calculated 
confusionMatrixTraining <- table(ypredicted, yTrain)
print(confusionMatrixTraining)
#Similar process is done for the test data
testSetN<-cbind(1,testSet)
z2<-sigmoidFunc(testSetN,wih)
YpredictedTest<-softmaxFunc(z2,vih)
ypredictedTest <- apply(YpredictedTest, MARGIN = 1, FUN = which.max)
confusionMatrixTest <- table(ypredictedTest, yTest)
print(confusionMatrixTest)
