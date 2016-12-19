library(psych)

data("USJudgeRatings")
pc <- principal(USJudgeRatings, nfactors = 1)
pc

# h2 is explaination level of pc1 to each variables, pc对每一个变量方差的解释度
# u2 is 1-h2, the part that cannot be explained by PC1

library(GPArotation)

# Now We have rotations
# 旋转会让成分的loadings变得更容易解释，对成分去燥
# varimax选择的目的是为了让每个成分只由几个变量来解释
# 下面rc显示得出，RC1主要由变量1-4解释，RC2主要由
# 变量5-8解释。这就是varimax旋转后的结果。
rc<-principal(Harman23.cor$cov,nfactors=2,rotate="varimax")
# 旋转前后的Cumulative Var没有变化，但是每个PC的variance变了。
# 所以从定义上讲，这里的PC已经不能被叫做PC了。
rc

# score是每个主成分的得分，而weight是variable相乘得到主成分的系数
pc<-principal(USJudgeRatings[,-1],nfactors=2,score=TRUE)
head(pc$scores)
# score是variable乘以weight之后相加得出的结果。可以理解为PC画图时候的坐标。

# EFA是用更少的潜在的几个潜在的变量来解释更多的可观察变量
# X_i = a1F1 + a2F2 + ... + apFp + U
# i from 1 to k, p << k
options(digits = 2)

covariances <- ability.cov$cov
correlations <- cov2cor(covariances)
correlations
# Determine number of factors we need
fa.parallel(correlations,n.obs=112,fa="both",n.iter=100,main="Scree plots
with parallel analysis")
fa<-fa(correlations,nfactors=2,rotate="none",fm="pa")
# fm决定了寻找factors的方法，pa是主轴迭代法。别的方法还有很多。
fa

fa.varimax<-fa(correlations,nfactors=2,rotate="varimax",fm="pa")
fa.varimax
# varimax强制两个factors不相关