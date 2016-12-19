# 预测模型的准确率可以用2种方法来提高：要么进行特征设计,要么直接使用boosting算法。
# Bagging：这是一种方法，当你使用随机采样的数据，建立学习算法，采取简单的手段以找到bagging的可能性。
# Boosting：与Bagging类似，但是，对样本的选择更智能。我们随后会对难以分类的样本分配较大的权重。

# 核心概念应该是给比较难分类的样本增加权重

# 在gbm包中，采用的是决策树作为基学习器，重要的参数设置如下： 

# 损失函数的形式(distribution)
# 迭代次数(n.trees)
# 学习速率(shrinkage)
# 再抽样比率(bag.fraction)
# 决策树的深度(interaction.depth)

#损失函数的形式容易设定，分类问题一般选择bernoulli分布，
#而回归问题可以选择gaussian分布。学习速率方面，我们都知道步子迈得
#太大容易扯着，所以学习速率是越小越好，但是步子太小的话，步数就得增加，
#也就是训练的迭代次数需要加大才能使模型达到最优，这样训练所需时间和
#计算资源也相应加大了。
#gbm作者的经验法则是设置shrinkage参数在0.01-0.001之间，
#而n.trees参数在3000-10000之间。

# Here we comparing two models approach
# gbm vs. glmnet. Boosting vs Regression.
library(caret)
options(digits = 4)
titanicDF <- read.csv('http://math.ucdenver.edu/RTutorial/titanic.txt',sep='\t')
print(str(titanicDF))
head(titanicDF)

# Using these variables to predict survival status
# Changing name columns to useful variables
titanicDF$Title <- ifelse(grepl('Mr ',titanicDF$Name),'Mr',
                          ifelse(grepl('Mrs ',titanicDF$Name),'Mrs',
                          ifelse(grepl('Miss',titanicDF$Name),'Miss',
                                 'Nothing'))) 
# Imputing missing values
titanicDF$Age[is.na(titanicDF$Age)] <- median(titanicDF$Age, na.rm=T)

# Extracting data for building models
titanicDF <- titanicDF[c('PClass', 'Age',    'Sex',   'Title', 'Survived')]
print(str(titanicDF))

# Create a dummy variable column from factor variables.If it contains
# 3 factors, then 3 new columns will be created.
titanicDF$Title <- as.factor(titanicDF$Title)
titanicDummy <- dummyVars("~.",data=titanicDF, fullRank=F)
titanicDF <- as.data.frame(predict(titanicDummy,titanicDF))
print(names(titanicDF))

# Show the proportion of survival
# This is an important step because if the proportion was smaller than 
# 15%, it would be considered a rare event and would be more challenging to model.
prop.table(table(titanicDF$Survived))

# Define names of variables
outcomeName <- 'Survived'
predictorsNames <- names(titanicDF)[names(titanicDF) != outcomeName]

titanicDF$Survived2 <- ifelse(titanicDF$Survived==1,'yes','nope')
titanicDF$Survived2 <- as.factor(titanicDF$Survived2)
outcomeName <- 'Survived2'

set.seed(1234)
splitIndex <- createDataPartition(titanicDF[,outcomeName], p = .75, list = FALSE, times = 1)
trainDF <- titanicDF[splitIndex,]
testDF  <- titanicDF[-splitIndex,]

# we’re going to cross-validate the data 3 times, 
# therefore training it 3 times on different portions of the data 
# before settling on the best tuning parameters (for gbm it is trees, 
# shrinkage, and interaction depth)
objControl <- trainControl(method='cv', number=3, returnResamp='none', 
                           summaryFunction = twoClassSummary, classProbs = TRUE)

# Metric here is ROC because we want classification of gbm
# instead of regression. If regression, we want RMSE.
objModel <- train(trainDF[,predictorsNames], trainDF[,outcomeName], 
                  method='gbm', 
                  trControl=objControl,  
                  metric = "ROC",
                  preProc = c("center", "scale"))

summary(objModel)
print(objModel)

predictions <- predict(object=objModel, testDF[,predictorsNames], type='raw')
head(predictions)
postResample(pred=predictions, obs=as.factor(testDF[,outcomeName]))

# Look at probabilities as result
library(pROC)
predictions <- predict(object=objModel, testDF[,predictorsNames], type='prob')
head(predictions)
auc <- roc(ifelse(testDF[,outcomeName]=="yes",1,0), predictions[[2]])
print(auc$auc)

###################### Generalized Linear Model ######################
getModelInfo()$glmnet$type

outcomeName <- 'Survived'

set.seed(1234)
splitIndex <- createDataPartition(titanicDF[,outcomeName], p = .75, list = FALSE, times = 1)
trainDF <- titanicDF[ splitIndex,]
testDF  <- titanicDF[-splitIndex,]

objControl <- trainControl(method='cv', number=3, returnResamp='none')
objModel <- train(trainDF[,predictorsNames], trainDF[,outcomeName], method='glmnet',  
                  metric = "RMSE", 
                  trControl=objControl)
predictions <- predict(object=objModel, testDF[,predictorsNames])
auc <- roc(testDF[,outcomeName], predictions)
print(auc$auc)

plot(varImp(objModel,scale=F))
