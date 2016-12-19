setwd("../兴趣")
train.col.types <- c('integer', # PassengerID
                     'factor', # Survived
                     'factor', # Pclass
                     'character', # Name
                     'factor', # Sex
                     'numeric', #Age
                     'integer', # SibSp
                     'integer', # Parch
                     'character', # Ticket
                     'numeric', # Fare
                     'character', # Cabin
                     'factor' # Embarked
                     )
test.col.types <- train.col.types[-2]
train.raw <- read.csv("train.csv", colClasses = train.col.types, na.strings = c("NA", ""))
test.raw <- read.csv("test.csv", colClasses = test.col.types, na.strings = c("NA", ""))
dim(train.raw)

summary(train.raw)

require(Amelia)
missmap(train.raw, main = "NA plot", col=c("yellow", "black"), legend=FALSE)
# We can see Cabin has a lot of missing values, Embarked only has 2 missing values

# Suvival vs. Died
barplot(table(train.raw$Survived), names.arg = c("Survived", "Died"), 
        main = "Survived vs. Died")

# Class level effects on survival
survive.rate.class <- table(train.raw$Survived, train.raw$Pclass)
barplot(survive.rate.class, names.arg = c("one", "two", "three"),
        main = "Passenger class vs. Survival",
        legend.text = c("die", "survival"),
        args.legend = list(x="topleft"))

round((survive.rate.class[2,]/colSums(survive.rate.class))*100, 2)

# Sex vs. Survival
survive.rate.class <- table(train.raw$Survived, train.raw$Sex)
barplot(survive.rate.class, names.arg = c("female", "male"),
        main = "Different Sex vs. Death",
        legend.text = c("die", "survival"))
round((survive.rate.class[2,]/colSums(survive.rate.class))*100, 2)

# Age vs. Survival
age.breaker <- c(0, 18, 50, 100)
age.cut <- cut(train.raw$Age, breaks=age.breaker,labels = c("child", "adult", "senior"))
train.raw$age.cut <- age.cut
survive.rate.class <- table(train.raw$Survived, train.raw$age.cut)
barplot(survive.rate.class, main = "age vs. death", legend.text = c("death", "survival"),
        args.legend = list(x="topleft"))
round((survive.rate.class[2,]/colSums(survive.rate.class))*100, 2)

# Mosaic Plot for visulization
mosaicplot(train.raw$Pclass ~ train.raw$Survived, shade = FALSE, color = TRUE,
           xlab = "class", ylab = "survival")


# Correlation Analysis
train.corrgram <- train.raw

train.corrgram$Survived <- as.numeric(train.corrgram$Survived)
train.corrgram$Pclass <- as.numeric(train.corrgram$Pclass)
train.corrgram$Embarked <- as.numeric(train.corrgram$Embarked)
train.corrgram$Sex <- as.numeric(train.corrgram$Sex)
train.corrgram[which(is.na(train.corrgram$Embarked)),]$Embarked=3
cor(train.corrgram[,c("Survived", "Pclass", "Embarked", "Sex", "Fare", "Age")])

# NA value analysis
require(corrgram)
corrgram.vars <- c("Survived", "Pclass", "Embarked", "Sex", "Fare", "Age",
                   "SibSp", "Parch", "Fare")
corrgram(train.corrgram[,corrgram.vars], lower.panel = panel.ellipse,
         upper.panel = panel.pie, text.panel = panel.txt, main = "Analysis")


# 建模
require(caret)
set.seed(0305)

intrain <- createDataPartition(train.raw$Survived, p = 0.8, list = FALSE)
training <- train.raw[intrain,]
testing <- train.raw[-intrain,]

# Use median to fill NA terms
first.class.age <- median(training[training$Pclass=="1",]$Age, na.rm = T)
second.class.age <- median(training[training$Pclass=="2",]$Age, na.rm = T)
third.class.age <- median(training[training$Pclass=="3",]$Age, na.rm = T)
training[is.na(training$Age)&training$Pclass=="1",]$Age <- first.class.age
training[is.na(training$Age)&training$Pclass=="2",]$Age <- second.class.age
training[is.na(training$Age)&training$Pclass=="3",]$Age <- third.class.age

third.class.fare <- median(training[training$Pclass=="3",]$Fare, na.rm = T)
training[is.na(training$Fare)&training$Pclass=="3",]$Fare <- third.class.fare

# Fit a logistic model
model.logit.1 <- train(Survived~Sex + Pclass + Age + Embarked + Fare,
                       data = training, method="glm")
model.logit.1
# Evaluate the logistic model
first.class.age <- median(testing[testing$Pclass=="1",]$Age, na.rm = T)
second.class.age <- median(testing[testing$Pclass=="2",]$Age, na.rm = T)
third.class.age <- median(testing[testing$Pclass=="3",]$Age, na.rm = T)
testing[is.na(testing$Age)&testing$Pclass=="1",]$Age <- first.class.age
testing[is.na(testing$Age)&testing$Pclass=="2",]$Age <- second.class.age
testing[is.na(testing$Age)&testing$Pclass=="3",]$Age <- third.class.age

predict.model.1 <- predict(model.logit.1, testing)
table(testing$Survived, predict.model.1)
sensitivity(testing$Survived, predict.model.1)
specificity(testing$Survived, predict.model.1)

require(gmodels)
CrossTable(testing$Survived, predict.model.1)

require(ROCR)

predictions.model.1 <- prediction(c(predict.model.1), labels = testing$Survived)
perf <- performance(predictions.model.1, measure = "tpr", x.measure = "fpr")
plot(perf, main =  "ROC curve", col = "blue", lwd = 2)
abline(a = 0, b = 1, lwd = 2, lty = 2)

# Replace NA in raw training for prediction
first.class.age <- median(test.raw[test.raw$Pclass=="1",]$Age, na.rm = T)
second.class.age <- median(test.raw[test.raw$Pclass=="2",]$Age, na.rm = T)
third.class.age <- median(test.raw[test.raw$Pclass=="3",]$Age, na.rm = T)
test.raw[is.na(test.raw$Age)&test.raw$Pclass=="1",]$Age <- first.class.age
test.raw[is.na(test.raw$Age)&test.raw$Pclass=="2",]$Age <- second.class.age
test.raw[is.na(test.raw$Age)&test.raw$Pclass=="3",]$Age <- third.class.age
test.raw[153,]$Fare <- third.class.fare

predict.final.model.1 <- predict(model.logit.1, newdata = test.raw)
predictions <- data.frame(PassergerID=test.raw$PassengerId, Survived=predict.final.model.1)
# 此结果可以提交到kaggle上, 这是final model

# 我们还有4个variable没有用， sibsp，parch，Cabin，names 
# 而且填补NA方式用median也不科学，我们可以尝试优化模型
train.col.types <- c('integer', # PassengerID
                     'factor', # Survived
                     'factor', # Pclass
                     'character', # Name
                     'factor', # Sex
                     'numeric', #Age
                     'integer', # SibSp
                     'integer', # Parch
                     'character', # Ticket
                     'numeric', # Fare
                     'character', # Cabin
                     'factor' # Embarked
)
test.col.types <- train.col.types[-2]
train.raw <- read.csv("train.csv", colClasses = train.col.types, na.strings = c("NA", ""))
test.raw <- read.csv("test.csv", colClasses = test.col.types, na.strings = c("NA", ""))

# 用title中的信息来进行数据的优化。类似Mr MISS Mrs等等
getTitle <- function(data) {
  title.start <- regexpr("\\, [A-Z]{1,20}\\.", data$Name, TRUE)
  title.end <- title.start + attr(title.start, "match.length")-1
  data$Title <- substr(data$Name, title.start+2, title.end-1)
  return(data$Title)
}

train.raw$Title <- getTitle(train.raw)

require(dplyr)
head(train.raw %>% group_by(Title)%>%summarise(count=n())%>%arrange(desc(count)))

# We only consider titles in these groups
title.filter <- c("Mr", "Mrs", "Miss", "Master", "Professional")

recodeTitle <- function(data, title, filter) {
  if(! (data %in% title.filter))
    data = "Professional"
  return(data)
}
train.raw$Title <- sapply(train.raw$Title, recodeTitle, title.filter)

# 更精确的NA补足
imputeAge <- function(Age, Title, title.filter) {
  for(v in title.filter) {
    Age[is.na(Age)] = median(Age[Title==v], na.rm = T)
  }
  return(Age)
}
title.filter <- c("Mr", "Mrs", "Miss", "Master", "Professional")
train.raw$Age <- imputeAge(train.raw$Age, train.raw$Title, title.filter)

imputeEmbarked <- function(Embarked) {
  Embarked[is.na(Embarked)] <- "S"
  return(Embarked)
}
train.raw$Embarked <- imputeEmbarked(train.raw$Embarked)

imputeFare <- function(fare, pclass, pclass.filter) {
  for(v in pclass.filter) {
    fare[is.na(fare)] <- median(fare[pclass==v], na.rm = T)
  }
  return(fare)
}
pclass.filter <- c(1,2,3)
train.raw$Fare <- imputeFare(train.raw$Fare, train.raw$Pclass, pclass.filter)

set.seed(920305)
intrain <- createDataPartition(train.raw$Survived, p = 0.8, list = FALSE)
training <- train.raw[intrain,]
testing <- train.raw[-intrain,]

model.logit.2 <- train(Survived~Sex+Pclass+Age+Embarked+Fare+Title+SibSp+
                         Parch, data = training, method="glm")
model.logit.2

predict.model.2 <- predict(model.logit.2, testing)
table(testing$Survived, predict.model.2)
sensitivity(testing$Survived, predict.model.2)
specificity(testing$Survived, predict.model.2)

# 预测kaggle挑战一发
title.filter <- c("Mr", "Mrs", "Miss", "Master", "Professional")
test.raw$Title <- getTitle(test.raw)
test.raw$Title <- sapply(test.raw$Title, recodeTitle, title.filter)
test.raw$Age <- imputeAge(test.raw$Age, test.raw$Title, title.filter)
test.raw$Embarked <- imputeEmbarked(test.raw$Embarked)
test.raw$Fare <- imputeFare(test.raw$Fare, test.raw$Pclass, pclass.filter)
predict.final.model.2 <- predict(model.logit.2, newdata = test.raw)
predictions <- data.frame(PassengerId=test.raw$PassengerId, Survived=predict.final.model.2)
write.csv(predictions, file = "Titanic_predictions_2.csv", row.names = F, quote=F)
