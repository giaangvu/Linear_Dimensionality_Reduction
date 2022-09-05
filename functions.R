#SOURCE FILE FOR USER DEFINED FUNCTIONS

#library
options(scipen = 999)
library(MASS) #lda and boston
library(stats)
library(dplyr)
library(pls) #pcr and pls
# install.packages("remotes")
# remotes::install_github("tsieger/tsiMisc")
library(tsiMisc) #Barshan method, https://tsieger.github.io/tsiMisc/reference/spca.html

#load data
data("Boston")
Ionos <- data.frame(read.table("ionosphere.data", header = F, sep = ",")) 
#Ref for Ionos: https://archive.ics.uci.edu/ml/datasets/ionosphere

##### PCR for continuous Y
my.pcr <- function(X, train, test){
  #make result matrix
  result <- data.frame(mse_train = rep(0,ncol(X)), mse_test = rep(0, ncol(X)))
  #rename last variable as "Y" to make the function work with other datasets
  train <- data.frame(train)
  colnames(train)[ncol(train)] <- "Y"
  test <- data.frame(test)
  colnames(test)[ncol(test)] <- "Y"
  #fit model and find MSE for train and test data, insert them into result
  m <- pcr(Y ~., data=data.frame(train), ncomp=ncol(X))
  resid <- matrix(m$residuals, ncol = ncol(X))
  for (i in 1:ncol(X)) {
    result[i,1] <- mean(resid[,i]^2) #mse train
    result[i,2] <- mean((data.frame(test)$Y - predict(m, data.frame(test), ncomp = i))^2) #mse test
  }
  return(result)
}

##### PLS for continuous Y
my.pls <- function(X, train, test){
  #make result matrix
  result <- data.frame(mse_train = rep(0,ncol(X)), mse_test = rep(0, ncol(X)))
  #rename last variable as "Y" to make the function work with other datasets
  train <- data.frame(train)
  colnames(train)[ncol(train)] <- "Y"
  test <- data.frame(test)
  colnames(test)[ncol(test)] <- "Y"
  #fit model and find MSE for train and test data, insert them into result
  m <- plsr(Y ~., data=data.frame(train), ncomp=ncol(X))
  resid <- matrix(m$residuals, ncol = ncol(X))
  for (i in 1:ncol(X)) {
    result[i,1] <- mean(resid[,i]^2) #mse train
    result[i,2] <- mean((data.frame(test)$Y - predict(m, data.frame(test), ncomp = i))^2) #mse test
  }
  return(result)
}

##### Barshan for continuous Y
my.Barshan <- function(X, Y, train, test){
  #get the matrix Q and it's eigen decomp (see Barshan paper)
  m <- tsiMisc::spca(X,Y, center = F, scale = F, retx = F, debug = F)
  #make result matrix
  result <- data.frame(mse_train = rep(0,ncol(X)), mse_test = rep(0, ncol(X)))
  #rename last variable as "Y" to make the function work with other datasets
  train <- data.frame(train)
  colnames(train)[ncol(train)] <- "Y"
  test <- data.frame(test)
  colnames(test)[ncol(test)] <- "Y"
  #the algorithm
  for (i in 1:ncol(X)) {
    U <- data.frame(m$vectors[,1:i]) #U is p*k (1x13)
    
    X.trans <- t(U) %*% t(X) 
    #reduce dim of X, t(X) is p*n matrix (13*379), transformed data is now k*n
    train.trans <- data.frame(X=t(X.trans), Y=Y)
    fit <- lm(Y ~ ., data = train.trans)
    result[i,1] <- mean(fit$residuals^2) #train mse
    
    X.test.trans <- t(t(U) %*% t(test[,-ncol(test)])) 
    #transformed X in test set (last column is the column containing Y) 
    test.trans <- data.frame(X = X.test.trans,Y = test[,ncol(test)])
    result[i,2] <- mean((data.frame(test)$Y - predict(fit, newdata = test.trans))^2) # test mse
  }
  return(result)
}

##### LDA
my.lda <- function(train, test){
  #make result matrix
  result <- data.frame(c("Training set", "Test set"),
                       c(nrow(train), nrow(test)),
                       c(ncol(train)-1, ncol(test)-1),
                       c(length(unique(train[,ncol(train)])),length(unique(train[,ncol(test)]))),
                       c(0,0)
                       )
  names(result) <- c("Data Set", "n", "r", "K", "LDA")
  #fit model
  m <- lda(Y~., data = train) #must format data sets to have a column with name "Y" for response
  p <- predict(m) #predict for train data set
  p.t <- predict(m, newdata = test) #predict for test dataset
  #store misclassification rate in result
  result$LDA <- c(round(mean(p$class != train$Y),3), round(mean(p.t$class != test$Y),3))
  return(result)
}

##### PLS for binary Y
bin.pls <- function(m, ncomp, train, test){ 
  #m is the already fitted pls model, ncomp is number of components to keep, decided by screeplot
  #make result matrix
  result <- data.frame(c("Training set", "Test set"),
                      c(nrow(train), nrow(test)),
                      c(ncol(train)-1, ncol(test)-1),
                      c(length(unique(train[,ncol(train)])),length(unique(train[,ncol(test)]))),
                      c(0,0)
  )
  names(result) <- c("Data Set", "n", "r", "K", "PLS")
  p <- predict(m, ncomp=ncomp) #train predict
  p <- ifelse(p>0.5,1,0) #use a simple threshold 0.5 for classification
  p.t <- predict(m, newdata = test, ncomp=ncomp)
  p.t <- ifelse(p.t>0.5,1,0) #test predict
  #store misclassification rate in result
  result$PLS <- c(round(mean(p != train$Y),3),round(mean(p.t != test$Y),3))
  return(result)
}

##### Barshan for binary Y
bin.Barshan <- function(ncomp, train, test){
  #make result matrix
  result <- data.frame(c("Training set", "Test set"),
                      c(nrow(train), nrow(test)),
                      c(ncol(train)-1, ncol(test)-1),
                      c(length(unique(train[,ncol(train)])),length(unique(train[,ncol(test)]))),
                      c(0,0)
  )
  names(result) <- c("Data Set", "n", "r", "K", "Barshan")
  m <- tsiMisc::spca(train[,-ncol(train)],train$Y, center = F, scale = F, retx = F, debug = F)
  U <- m$vectors[,1:ncomp]
  X.train.trans <- t(U) %*% t(train[,-ncol(train)]) #reduce dim of X, t(X) is p*n matrix (33*263), transformed data is now k*n (4*263)
  X.test.trans <- t(U) %*% t(test[,-ncol(test)])
  train.trans <- data.frame(X=t(X.train.trans), Y=train$Y)
  test.trans <- data.frame(X=t(X.test.trans), Y=test$Y)
  
  #logistic
  fit <- glm(Y~., data = train.trans, family = binomial(link="logit"))
  #again use threshold prediction > 0.5 -> class 1, otherwise class 2
  p <-  ifelse(fit$fitted.values > 0.5, 1, 0)
  
  #prediction with test data
  p.t <- predict(fit, newdata = test.trans, type = "response")
  p.t1 <- ifelse(p.t > 0.5,1,0)
  
  #store misclassification rate in result
  result$Barshan <- c(round(mean(p != train.trans$Y),3),round(mean(p.t1 != test.trans$Y),3))
  return(result)
}
