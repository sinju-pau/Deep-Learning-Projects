
library(caTools)
library(ggplot2)
library(GGally)

# Importing the dataset
custchurn <- read.csv(url("http://www.sgi.com/tech/mlc/db/churn.data"),header =FALSE)

head(custchurn,10)

sum(is.na(custchurn))

colnames(custchurn) <- c("state","account_length","area_code","phone_number","international_plan","voice_mail_plan","number_vmail_messages","total_day_minutes","total_day_calls","total_day_charge","total_eve_minutes","total_eve_calls","total_eve_charge","total_night_minutes","total_night_calls","total_night_charge","total_intl_minutes","total_intl_calls","total_intl_charge","number_customer_service_calls","churned")

custchurn

custchurn <- custchurn[,c(-1,-3,-4)]
# Do a random shuffle to the dataset
custchurn <- custchurn[sample(1:nrow(custchurn)),]
head(custchurn,10)
str(custchurn)

custchurn$international_plan <- as.numeric(custchurn$international_plan)
custchurn$voice_mail_plan <- as.numeric(custchurn$voice_mail_plan)
custchurn$churned <- as.numeric(custchurn$churned)
custchurn$churned <- ifelse(custchurn$churned == 2,1,0)
str(custchurn)

ggpairs(custchurn, columns = c(1,4,5,6,7,8,9,10,11,12,13,14,15,16,17), upper = "blank", aes(colour = churned, alpha = 0.8))

set.seed(123)
split <- sample.split(custchurn$churned, SplitRatio = 0.8)
training_set <- subset(custchurn, split == TRUE)
test_set <- subset(custchurn, split == FALSE)

# Feature Scaling must to do in Deep Learning
training_set[,c(-2,-3,-18)] <- scale(training_set[,c(-2,-3,-18)])
test_set[,c(-2,-3,-18)] <- scale(test_set[,c(-2,-3,-18)])
head(training_set,5)
head(test_set,5)

library(h2o)
h2o.init(nthreads = -1)

# Building the classifier
classifier <- h2o.deeplearning(y = 'churned',
                         training_frame = as.h2o(training_set),
                         activation = 'Rectifier',
                         hidden = c(6,6),
                         epochs = 100,
                         train_samples_per_iteration = -2)

summary(classifier)

plot(classifier)

# Predicting the Test set results
y_pred = h2o.predict(classifier, newdata = as.h2o(test_set[-18]))
y_pred = (y_pred > 0.5) 
y_pred = as.vector(y_pred) #converting h2o object, y_pred, to vector

# Making the Confusion Matrix
cm <- table(test_set[,18], y_pred)
cm

h2o.shutdown()

Accuracy <- sum(diag(cm))/nrow(test_set)
Accuracy


