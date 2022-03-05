##Preprocess
#Read data
df <- read.csv("ppp_model2.csv")
#Remove high correlation data
df <- df[,c(-15,-16)]
#Check data types
for (col in colnames(df))
{cat(col); cat(str(df[[col]]))}

#Convert data to correct format
df_lg <- df
for (col in colnames(df_lg[,c(2,5,6,7,8,9,10,11,12,14,15,16)]))
  {df_lg[[col]] <- factor(df_lg[[col]])}
df_nb <- df
for (col in colnames(df_nb[,c(-1,-3,-4,-17)]))
  {df_nb[[col]] <- factor(df_nb[[col]])}

#Split the data
library("caret")
set.seed(1)
train <- createDataPartition(df$is_removed, p=0.7, list=FALSE)
df_lg_train <- data.frame(df_lg[train,])
df_lg_test <- data.frame(df_lg[-train,])
df_nb_train <- data.frame(df_nb[train,])
df_nb_test <- data.frame(df_nb[-train,])

#Adding standard weights to deal with imbalanced classes
stdw1 <- (575559/(2*25751))
stdw0 <- (575559/(2*549808))

##Logistic model
#Assign standard weights
df_lg_train$wt <- ifelse(df_lg_train$is_removed==1, stdw1, stdw0)
df_lg_test$wt <- ifelse(df_lg_test$is_removed==1, stdw1, stdw0)

#Logistic model & predictions
fit1 <- glm(is_removed~.-wt, data=df_lg_train, family="binomial", weights=wt)
cutoff1 <- 0.5
actual_train <- df_lg_train$is_removed
actual_train <- ifelse(actual_train==0, "Unremoved","Removed")
actual_train <- factor(actual_train, levels=c("Unremoved","Removed"))
predicted_train_p <- predict(fit1, data=df_lg_train[,-13], type="response")
predicted_train <- ifelse(predicted_train_p > cutoff1, "Removed","Unremoved")
predicted_train <- factor(predicted_train, levels=c("Unremoved","Removed"))
actual_test <- df_lg_test$is_removed
actual_test <- ifelse(actual_test==0, "Unremoved","Removed")
actual_test <- factor(actual_test, levels=c("Unremoved","Removed"))
predicted_test_p <- predict(fit1, newdata=df_lg_test[,-13], type="response")
predicted_test <- ifelse(predicted_test_p > cutoff1, "Removed","Unremoved")
predicted_test <- factor(predicted_test, levels=c("Unremoved","Removed"))

#Confusion & error rate for train data
lg_cm1 <- table(actual_train, predicted_train)
(ER1 <- (lg_cm1[1,2]+lg_cm1[2,1])/sum(lg_cm1))
#Confusion & error rate for test data
lg_cm2 <- table(actual_test, predicted_test)
(ER2 <- (lg_cm2[1,2]+lg_cm2[2,1])/sum(lg_cm2))

#Performance metrics
library(knitr)
sen_train <- sum(predicted_train == "Removed" & actual_train == "Removed")/sum(actual_train == "Removed")
spe_train <- sum(predicted_train == "Unremoved" & actual_train == "Unremoved")/sum(actual_train == "Unremoved")
ppv_train <- sum(predicted_train == "Removed" & actual_train == "Removed")/sum(predicted_train == "Removed")
npv_train <- sum(predicted_train == "Unremoved" & actual_train == "Unremoved")/sum(predicted_train == "Unremoved")
sen_test <- sum(predicted_test == "Removed" & actual_test == "Removed")/sum(actual_test == "Removed")
spe_test <- sum(predicted_test == "Unremoved" & actual_test == "Unremoved")/sum(actual_test == "Unremoved")
ppv_test <- sum(predicted_test == "Removed" & actual_test == "Removed")/sum(predicted_test == "Removed")
npv_test <- sum(predicted_test == "Unremoved" & actual_test == "Unremoved")/sum(predicted_test == "Unremoved")
metrics <-c("sen_train", "spe_train", "ppv_train", "npv_train", "sen_test", "spe_test", "ppv_test", "npv_test")
values <- c(sen_train, spe_train, ppv_train, npv_train, sen_test, spe_test, ppv_test, npv_test)
kable(data.frame(metrics, values))

#Testing different Weights & cutoffs
#ER1 <- rep(0,9)
ER2 <- rep(0,9)
ER3 <- rep(0,9)
ER4 <- rep(0,9)

w <- c(8,10,12)
c <- c(0.4,0.5,0.6)
for (j in c){
  for (i in w){
    df_lg_train$wt <- ifelse(df_lg_train$is_removed==1, i, 1)
    df_lg_test$wt <- ifelse(df_lg_test$is_removed==1, i, 1)
    #Logistic model & predictions
    fit1 <- glm(is_removed~.-wt, data=df_lg_train, family="binomial", weights=wt)
    #actual_train <- df_lg_train$is_removed
    #actual_train <- ifelse(actual_train==0, "Unremoved","Removed")
    #actual_train <- factor(actual_train, levels=c("Unremoved","Removed"))
    #predicted_train_p <- predict(fit1, data=df_lg_train[,-13], type="response")
    #predicted_train <- ifelse(predicted_train_p > j, "Removed","Unremoved")
    #predicted_train <- factor(predicted_train, levels=c("Unremoved","Removed"))
    actual_test <- df_lg_test$is_removed
    actual_test <- ifelse(actual_test==0, "Unremoved","Removed")
    actual_test <- factor(actual_test, levels=c("Unremoved","Removed"))
    predicted_test_p <- predict(fit1, newdata=df_lg_test[,-13], type="response")
    predicted_test <- ifelse(predicted_test_p > j, "Removed","Unremoved")
    predicted_test <- factor(predicted_test, levels=c("Unremoved","Removed"))
    
    x <- (i/2-3)+(3*(j*10-4))
    #Confusion & error rate for train data
    #lg_cm1 <- table(actual_train, predicted_train)
    #(ER1[x] <- (lg_cm1[1,2]+lg_cm1[2,1])/sum(lg_cm1))
    #Error rate, sensitivity, specificity for test data
    lg_cm2 <- table(actual_test, predicted_test)
    ER2[x] <- (lg_cm2[1,2]+lg_cm2[2,1])/sum(lg_cm2)
    ER3[x] <- (lg_cm2[2,2]/(lg_cm2[2,1]+lg_cm2[2,2]))
    ER4[x] <- (lg_cm2[1,1]/(lg_cm2[1,1]+lg_cm2[1,2]))
    #Counting loop
    print(x)
  }
}
#Metrics
ER1
ER2
ER3
ER4

##Naive Bayes model
##Assign standard weights
df_nb_train$wt <- ifelse(df_nb_train$is_removed==1, stdw1, stdw0)
df_nb_test$wt <- ifelse(df_nb_test$is_removed==1, stdw1, stdw0)

#Fit model
library(e1071)
nb_model <- naiveBayes(is_removed~.-wt, data=df_nb_train, weights=wt)
nb_model

#Predict on train data
nb_predicted.probability <- predict(nb_model, newdata = df_nb_train[,-13], type="raw")
cutoff1 <- 0.5
actual_train2 <- df_nb_train$is_removed
actual_train2 <- ifelse(actual_train2==0, "Unremoved","Removed")
actual_train2 <- factor(actual_train2, levels=c("Unremoved","Removed"))
nb_predicted.probability <- predict(nb_model, newdata = df_nb_train[,-13], type="raw")
nb_predicted.probability <- nb_predicted.probability[,2]
predicted_train2 <- ifelse(nb_predicted.probability > cutoff1, "Removed","Unremoved")
predicted_train2 <- factor(predicted_train2, levels=c("Unremoved","Removed"))
#Predict on test data
actual_test2 <- df_nb_test$is_removed
actual_test2 <- ifelse(actual_test2==0, "Unremoved","Removed")
actual_test2 <- factor(actual_test2, levels=c("Unremoved","Removed"))
nb_predicted.probability2 <- predict(nb_model, newdata = df_nb_test[,-13], type="raw")
nb_predicted.probability2 <- nb_predicted.probability2[,2]
predicted_test2 <- ifelse(nb_predicted.probability2 > cutoff1, "Removed","Unremoved")
predicted_test2 <- factor(predicted_test2, levels=c("Unremoved","Removed"))
#confusion matrix & error rate on train data
nb_cm1 <- table(df_nb_train$is_removed, predicted_train2, dnn=list('actual','predicted'))
(nb_er1 <- (nb_cm1[1,2]+nb_cm1[2,1])/sum(nb_cm1))
#confusion matrix & error rate on test data
nb_cm2 <- table(df_nb_test$is_removed, predicted_test2, dnn=list('actual','predicted'))
(nb_er2 <- (nb_cm2[1,2]+nb_cm2[2,1])/sum(nb_cm2))

#Performance metrics
library(knitr)
sen_train2 <- sum(predicted_train2 == "Removed" & actual_train2 == "Removed")/sum(actual_train2 == "Removed")
spe_train2 <- sum(predicted_train2 == "Unremoved" & actual_train2 == "Unremoved")/sum(actual_train2 == "Unremoved")
ppv_train2 <- sum(predicted_train2 == "Removed" & actual_train2 == "Removed")/sum(predicted_train2 == "Removed")
npv_train2 <- sum(predicted_train2 == "Unremoved" & actual_train2 == "Unremoved")/sum(predicted_train2 == "Unremoved")
sen_test2 <- sum(predicted_test2 == "Removed" & actual_test2 == "Removed")/sum(actual_test2 == "Removed")
spe_test2 <- sum(predicted_test2 == "Unremoved" & actual_test2 == "Unremoved")/sum(actual_test2 == "Unremoved")
ppv_test2 <- sum(predicted_test2 == "Removed" & actual_test2 == "Removed")/sum(predicted_test2 == "Removed")
npv_test2 <- sum(predicted_test2 == "Unremoved" & actual_test2 == "Unremoved")/sum(predicted_test2 == "Unremoved")
metrics <-c("sen_train", "spe_train", "ppv_train", "npv_train", "sen_test", "spe_test", "ppv_test", "npv_test")
values <- c(sen_train2, spe_train2, ppv_train2, npv_train2, sen_test2, spe_test2, ppv_test2, npv_test2)
kable(data.frame(metrics, values))

#Testing different Weights
ER5 <- rep(0,18)
ER6 <- rep(0,18)
ER7 <- rep(0,18)
ER8 <- rep(0,18)

w1 <- c(5,10,15,20,25,30)
c1 <- c(0.4, 0.5, 0.6)
for (j in c1){
  for (i in w1){
    df_nb_train$wt <- ifelse(df_nb_train$is_removed==1, i, 1)
    df_nb_test$wt <- ifelse(df_nb_test$is_removed==1, i, 1)
    
    #Fit model
    library(e1071)
    nb_model <- naiveBayes(is_removed~.-wt, data=df_nb_train, weights=wt)
    nb_model
    #Predict on train data
    nb_predicted.probability <- predict(nb_model, newdata = df_nb_train[,-13], type="raw")
    actual_train2 <- df_nb_train$is_removed
    actual_train2 <- ifelse(actual_train2==0, "Unremoved","Removed")
    actual_train2 <- factor(actual_train2, levels=c("Unremoved","Removed"))
    nb_predicted.probability <- predict(nb_model, newdata = df_nb_train[,-13], type="raw")
    nb_predicted.probability <- nb_predicted.probability[,2]
    predicted_train2 <- ifelse(nb_predicted.probability > j, "Removed","Unremoved")
    predicted_train2 <- factor(predicted_train2, levels=c("Unremoved","Removed"))
    #Predict on test data
    actual_test2 <- df_nb_test$is_removed
    actual_test2 <- ifelse(actual_test2==0, "Unremoved","Removed")
    actual_test2 <- factor(actual_test2, levels=c("Unremoved","Removed"))
    nb_predicted.probability2 <- predict(nb_model, newdata = df_nb_test[,-13], type="raw")
    nb_predicted.probability2 <- nb_predicted.probability2[,2]
    predicted_test2 <- ifelse(nb_predicted.probability2 > j, "Removed","Unremoved")
    predicted_test2 <- factor(predicted_test2, levels=c("Unremoved","Removed"))
    
    x <- (i/5)+(6*(j*10-4))
    #Confusion & error rate for train data
    nb_cm1 <- table(actual_train2, predicted_train2)
    (ER5[x] <- (nb_cm1[1,2]+nb_cm1[2,1])/sum(nb_cm1))
    #Error rate, sensitivity, specificity for test data
    nb_cm2 <- table(actual_test2, predicted_test2)
    ER6[x] <- (nb_cm2[1,2]+nb_cm2[2,1])/sum(nb_cm2)
    ER7[x] <- (nb_cm2[2,2]/(nb_cm2[2,1]+nb_cm2[2,2]))
    ER8[x] <- (nb_cm2[1,1]/(nb_cm2[1,1]+nb_cm2[1,2]))
    print(x)
  }
}
#Metrics
ER5
ER6
ER7
ER8