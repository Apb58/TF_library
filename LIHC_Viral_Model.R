# Viral Load Multinomial Model in LIHC
# 7/24/18

# Working with two datasets:
# 1) Viral load (as measured as viral RPKM) in human HCC samples from TCGA
# 2) Clinical data from TCGA HCC samples, including survival status, age, tumor stage

library(ggplot2)
library(plyr)
library(reshape2)
library(glmnet)

# I want to know what the relationship is between viral load of HBV or HPV and tumor stage, and is viral load predictive? 
# Also, collect the total viral load per patient and see if that is predictive.

# Read in data files:
vdf <- read.delim('~/TF_library/data/TCGA-LIHC-Viral_Load.csv', header=T, sep=',', stringsAsFactors = F)
vdf <- vdf[which(vdf$Cancer == "LIHC"),]
clin <-read.delim('~/TF_library/data/clinical.tsv', header=T, sep='\t', stringsAsFactors = F)

# Simplfy the vdf sample IDs:
vdf$Sample <- substr(vdf$Sample,1,12)

# Merge the dfs:
colnames(clin)[2] <- "Sample" # To allow merge on Sample ID
tdf <- merge(vdf,clin, all.x = T)
tdf <- tdf[-which(tdf$tumor_stage == "not reported"),]

# Calculate the total 'viral load' of each sample:
tdf$total_viral_load <- apply(tdf[,3:29], 1, sum)

# Look at the distribution of viral loads:
ggplot(data=tdf, aes(y=total_viral_load, x=tumor_stage)) + geom_violin()

# Tons of outliers, should log-normalize and mean-center:
tdf$norm_viral_load <- scale(log(tdf$total_viral_load + 0.5), scale=F)
ggplot(data=tdf, aes(y=norm_viral_load, x=tumor_stage)) + geom_violin()
# These look a little better

# Simplfy stages:
tdf$tumor_stage <- replace(tdf$tumor_stage, which(tdf$tumor_stage == 'stage iiia' | tdf$tumor_stage == 'stage iiib' | tdf$tumor_stage == 'stage iiic'), "stage iii")
tdf$tumor_stage <- replace(tdf$tumor_stage, which(tdf$tumor_stage == 'stage iva' | tdf$tumor_stage == 'stage ivb'), "stage iv")
tdf$SimpleStageNumeric <- as.numeric(factor(tdf$tumor_stage))


# setup models with training subset of data:
lihc_mat <- cbind(log(tdf$HBV + 0.5), tdf$norm_viral_load, as.numeric(tdf$age_at_diagnosis))
rem <- which(tdf$age_at_diagnosis < 0)
lihc_mat <- lihc_mat[-rem,] # Remove those with no age data
response <- tdf[-rem,]$tumor_stage 
colnames(lihc_mat) <- c("HBV_Load", "Vir_Load", "Diag.Age")


# Create training and testing subset:
trn_sub <- sample(nrow(lihc_mat), 200, replace=F)
lihc_mat_trn <- lihc_mat[trn_sub,]
response_trn <- response[trn_sub]

lihc_mat_test <- lihc_mat[-trn_sub,]
true_response <- response[-trn_sub]


# Fit the Multinomial model using the training data:
lihc_vir_fit <- glmnet(x=lihc_mat_trn, y=response_trn, family = "multinomial", alpha=0.6)

# Perform cross-validation to determine the correct lambda of m.s.e to extract term coefficents for:
lmins <- NULL
for(i in 1:100){
  lihc_vir_cv.fit <- cv.glmnet(x=lihc_mat_trn, y=response_trn, family = "multinomial", nfolds=5, parallel = T) # 20-fold CV to calc. lambda
  lmins <- c(lmins, lihc_vir_cv.fit$lambda.min)
}
# Calc lmin as average from runs:
lmin <- mean(lmins)
# Get coefficents at this lambda:
coef(lihc_vir_fit, s=lmin)

## See if the model can predict the correct stages for the testing data:
test <- predict(lihc_vir_fit, newx=lihc_mat_test, s=lmin, type="class")
predict_acc <- sum(test == true_response, na.rm=T)/length(true_response)
predict_acc

# A relatively weak 58.1% accuracy (Just over a coin flip)




