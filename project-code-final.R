# 1) DATA SETUP ---------------------------------------------------------------------------
dirpath = "/Users/carolinejung/Library/Mobile Documents/com~apple~CloudDocs/Desktop/8th sem SPRING 2025/DS 340H/Datasets/bluebikedata/"
df = read.csv(paste0(dirpath,"bluebikes_finalsubsetv1.csv"), header=TRUE)

# turn categorical variables into factors & relevel multi-class variables
colnames(df)[c(2,6,8,9,11)]
df[,c(2,9,11)] = lapply(df[,c(2,9,11)], FUN=as.factor) # binary 

df$month = factor(df$month, levels=c("1","2","3","4","5","6","7","8","9","10","11","12"))
df$started_time_of_day = factor(df$started_time_of_day, levels=c("morning", "afternoon", "evening", "night"))
df$ended_time_of_day = factor(df$ended_time_of_day, levels=c("morning", "afternoon", "evening", "night"))

# total docks - change integer types to numeric
df[,c(12,13)] = lapply(df[,c(12,13)], FUN=as.numeric)

# add new column - logged trip length value
df$trip_length_log = log(df$trip_length)

# change infinite values to 0 (infinite values caused by original trip length being 0)
df$trip_length_log[is.infinite(df$trip_length_log)] = 0

# remove unnecessary columns
colnames(df)
df = df[,-c(1,3:8,14:16)] 
colnames(df)

table(df$member_casual) # class imbalance
barplot(table(df$member_casual), main="Distribution of Bike Trips 
        Taken by Casual vs Members", ylab="Number of Bike Trips")

# SPLIT TRAIN & VALIDATION DATA ---------------------------------------------------------
set.seed(1)
train.index <- sample(1:nrow(df), size=0.7*nrow(df), replace=FALSE)
train <- df[train.index,]
val <- df[-train.index,]

# 2) MULTICOLLINEARITY ------------------------------------------------------------------
library(usdm)
x.var <- df[,-c(1:4,7,8)] # remove categorical and response variable
vifstep(x.var, th=5) # nothing to remove

# 3) MODEL & VARIABLE SELECTION ---------------------------------------------------------
# a) FIRST ORDER MODELS -----------------------------------------------------------------
# i) compare first-order models among diff thresholds with 10-fold cv
# set up folds for cv
n <- nrow(df)
num_features <- ncol(df)-1
K <- 10 
n.fold <- round(n/K) # size of each fold

set.seed(2342)
shuffle <- sample(1:n, n, replace=FALSE)
index.fold <- list()
for (i in 1:K) {
  if (i < K) {
    index.fold[[i]] <- shuffle[((i-1) * n.fold + 1) : (i * n.fold)]
  } else {
    index.fold[[i]] <- shuffle[((K-1) * n.fold + 1) : n]
  }
}

# compare among thresholds
thresholds = c(0.8, 0.7, 0.6, 0.5, 0.4)
num_t = length(thresholds)
thres_data_fmeas = data.frame(log_AIC=rep(NA,num_t), log_BIC=rep(NA,num_t),
                              prob_AIC=rep(NA,num_t), prob_BIC=rep(NA,num_t))
thres_data_diff = data.frame(log_AIC=rep(NA,num_t), log_BIC=rep(NA,num_t),
                             prob_AIC=rep(NA,num_t), prob_BIC=rep(NA,num_t))

# for each threshold
for (t in 1:num_t){
  threshold=thresholds[t]
  print(threshold)

  fold_data_fmeas = data.frame(log_AIC=rep(NA,K), log_BIC=rep(NA,K),
                               prob_AIC=rep(NA,K), prob_BIC=rep(NA,K))
  
  fold_data_diff = data.frame(log_AIC=rep(NA,K), log_BIC=rep(NA,K),
                              prob_AIC=rep(NA,K), prob_BIC=rep(NA,K))
  
  # for each fold (cv)
  for(f in 1:K) {
    print("fold"); print(f)
    
    # define first order models (ok bc set seed)
    logit = glm(formula=relevel(member_casual, "casual")~.,family=binomial(link="logit"), data=train[-index.fold[[f]],])
    logit_AIC = step(logit, direct="both", k=2, trace=FALSE)
    logit_BIC = step(logit, direct='both', k=log(n), trace=FALSE)
    
    probit = glm(formula=relevel(member_casual, "casual")~.,family=binomial(link="probit"), data=train[-index.fold[[f]],])
    probit_AIC = step(probit, direct="both", k=2, trace=FALSE)
    probit_BIC = step(probit, direct='both', k=log(n), trace=FALSE)
    
    models = list(logit_AIC, logit_BIC, probit_AIC, probit_BIC)
    f_meas = rep(NA,4)
    diff = rep(NA,4)
    j=1
    
    # for each model
    for (model in models){
      print("model")
      phat = predict(model, type="response", newdata=val)
      yhat = ifelse(phat>threshold, "member", "casual") # predictions
      
      # check to make sure the model predicts both member & casual levels
      if (length(unique(yhat)) == 2){
        conf = table(yhat, val$member_casual)
        sens = conf[2,2] / sum(conf[,2])
        prec = conf[2,2] / sum(conf[2,])
        spec = conf[1,1] / sum(conf[,1])
        
        f_meas[j] = (2*sens*prec)/(sens+prec)
        diff[j] = abs(sens-spec)
        
      } else { #conf matrix is not 2x2, cannot compute metrics
        f_meas[j] = NA
        diff[j] = NA
      }
      j = j+1 # update index of model
    }
    # record the f measures & diffs for this fold
    fold_data_fmeas[f,] = f_meas
    fold_data_diff[f,] = diff
  }
  # avg across folds & record for this threshold
  thres_data_fmeas[t,] = colMeans(fold_data_fmeas)
  thres_data_diff[t,] = colMeans(fold_data_diff)
}

thres_data_fmeas$threshold = thresholds
thres_data_diff$threshold = thresholds

# performance metrics (f measure and difference) based on threshold
thres_data_fmeas
thres_data_diff

# ii) best first-order model
summary(logit_AIC)

# b) HIGHER ORDER MODELS --------------------------------------------------------------
# i) tree ensemble method: bagging & random forest
library(randomForest)
k = ncol(df)-1 # number of features

# ii) support vector machines
# tuning for best parameters
library(e1071)
poly_best = tune(svm, member_casual~., data=train, kernel="polynomial", ranges=list(degree=3, cost = c(0.1,1,5,10)))
lin_best = tune(svm, member_casual~., data=train, kernel="linear", ranges=list(cost = c(0.1,1,5,10)))
rad_best = tune(svm, member_casual~., data=train, kernel="radial", ranges=list(cost = c(0.1,1,5,10)))
sig_best = tune(svm, member_casual~., data=train, kernel="sigmoid", ranges=list(cost = c(0.1,1,5,10)))

c(poly_best$best.performance, lin_best$best.performance, rad_best$best.performance, sig_best$best.performance) # polynomial has best score (want to minimize)
poly_best$best.parameters # degree 3, cost 5 --> use these best parameters

# iii) regression interaction terms
logit_AIC_int = step(logit, .~.^2, direct="both", k=2, trace=FALSE) 
logit_BIC_int = step(logit, .~.^2, direct='both', k=log(n), trace=FALSE)
probit_AIC_int = step(probit, .~.^2, direct="both", k=2, trace=FALSE)
probit_BIC_int = step(probit, .~.^2, direct="both", k=log(n), trace=FALSE)
# logit BIC interaction model was the only model to converge


# 4) CHOOSE FINAL MODEL USING CV ----------------------------------------------------
# note: use the same folds from above (first-order threshold tuning)
# check that there are members & casual in each fold
for(i in 1:K) {
  print(table(train[-index.fold[[i]],]$member_casual))
}

# save fold values here
fold_data_fmeas = data.frame(log_AIC_first=rep(NA,K), log_BIC_interact=rep(NA,K),
                       tree_bag=rep(NA,K), tree_rf=rep(NA,K), svm=rep(NA,K))
fold_data_diff = data.frame(log_AIC_first=rep(NA,K), log_BIC_interact=rep(NA,K),
                             tree_bag=rep(NA,K), tree_rf=rep(NA,K), svm=rep(NA,K))

# for each fold
for(i in 1:K) {
  # DEFINE ALL POSSIBLE MODELS (BEST FIRST & HIGHER ORDER)
  # best first order
  logit_AIC_first = glm(formula = relevel(member_casual, "casual") ~ rideable_type + 
                          month + round_trip + start_station_total_docks + end_station_total_docks + 
                          started_time_of_day + ended_time_of_day + trip_length_log, 
                        family = binomial(link = "logit"), data = train[-index.fold[[i]],])
  
  # log BIC interaction
  log_BIC_interact = glm(formula = relevel(member_casual, "casual") ~ rideable_type + 
                           round_trip + start_station_total_docks + end_station_total_docks + 
                           ended_time_of_day + trip_length_log + rideable_type:trip_length_log + 
                           round_trip:trip_length_log, family = binomial(link = "logit"), 
                         data = train[-index.fold[[i]],])
  
  # tree bagging
  tree_bag = randomForest(member_casual~., data=train[-index.fold[[i]],], mtry=num_features)
  
  # tree random forest
  tree_rf = randomForest(member_casual~., data=train[-index.fold[[i]],], mtry=sqrt(num_features))
  
  # svm
  svm_best = svm(member_casual~., data=train[-index.fold[[i]],], type="C-classification", kernel="polynomial", cost=5, probability = TRUE)
  
  # compare models
  models = list(logit_AIC_first, log_BIC_interact, tree_bag, tree_rf, svm_best)
  f_meas = rep(NA, 5)
  diff = rep(NA,5)
  j = 1 # index for which model we are on
  
  # for each model
  for (model in models){
    if ("randomForest" %in% class(model)){
      print("tree")
      phat = predict(model, type="prob", newdata=val)[,2]
    } else if ("svm" %in% class(model)){
      print("svm")
      preds = predict(model, newdata=val, probability=TRUE)
      phat = attr(preds, "probabilities")[,1] # prob for member
    } else {
      print("regression")
      phat = predict(model, type="response", newdata=val)
    }

    # use the tuned threshold (0.8)
    yhat = ifelse(phat>0.8, "member", "casual")
    
    # check to make sure the model predicts both member & casual levels
    if (length(unique(yhat)) == 2){
      conf = table(yhat, val$member_casual)
      sens = conf[2,2] / sum(conf[,2])
      prec = conf[2,2] / sum(conf[2,])
      spec = conf[1,1] / sum(conf[,1])
      
      f_meas[j] = (2*sens*prec)/(sens+prec)
      diff[j] = abs(sens-spec)
      
    } else { #conf matrix is not 2x2, cannot compute metrics
      f_meas[j] = NA
      diff[j] = NA
    }
    j = j+1
  }
  fold_data_fmeas[i,] = f_meas
  fold_data_diff[i,] = diff
}
# avg over all folds to get f measures (col 1) & difference (col 2)
final_summary = rbind(colMeans(fold_data_fmeas), colMeans(fold_data_diff))
final_summary

# 5) DIAGNOSTICS -----------------------------------------------------------------
# logistic AIC 
# diagnostic plots
par(mfrow=c(2,2))
plot(logit_AIC) # not normal but okay
par(mfrow=c(1,1))

# outliers
df[c(699, 391, 347, 3411, 3224, 1559),] # none of them look too out of place

library('blorr')
blr_plot_diag_difdev(logit_AIC) # delta deviance for influential points

# 6) FINAL MODEL ------------------------------------------------------------------
# a) logistic AIC
best_reg = glm(formula=relevel(member_casual, "casual") ~ rideable_type + 
                   month + round_trip + start_station_total_docks + end_station_total_docks + 
                   started_time_of_day + ended_time_of_day + trip_length_log,
                 family=binomial(link="logit"), data=df) # on full data
summary(best_reg)


# b) tree bagging
tree_bag = randomForest(member_casual~., data=df, mtry=num_features)
plot(tree_bag) # lowest around ntree 100
importance(tree_bag) # importance plot


