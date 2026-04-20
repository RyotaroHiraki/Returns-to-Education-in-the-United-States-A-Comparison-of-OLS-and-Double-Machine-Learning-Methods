#####################
####Developed ver####
#####################

setwd('/Users/ryotarohiraki/Desktop/Spring2025/Economics of Education/final paper(35%)/data')

library(haven)       # データ読み込み
library(DoubleML)    # Double Machine Learning
library(mlr3)        # 機械学習モデル
library(mlr3learners) 
library(data.table)  # データ管理
library(sandwich)    # ロバスト標準誤差
library(lmtest)      # 回帰結果のテスト
library(ranger)
library(tidyverse)
library(dplyr)

########################
###data cleaning process
########################

#read a personal dataset
raw_cps2024 <- read_csv("asecpub24csv/pppub24.csv",col_types = cols(PERIDNUM = col_character())) |> zap_formats()

#read a household dataset
raw_hcps2024 <- read_csv("asecpub24csv/hhpub24.csv", col_types = cols(H_IDNUM = col_character())) |> zap_formats()

#avoid ids to be a exponential form by importing as character

#GESTFIPS- state code (ONLY EXISTS IN HOUSEHOLD SURVEY)

#A_HGA- educational attainment
#A_AGE- age
#A_SEX- sex(1-male, 2-female)
#A_MARITL- martial status
#PRDTRACE- race
#A_HRSPAY- hourly wage
#PERIDNUM- 22 digits unique person identifier number.
#A_ERNLWT- earn weight (two implied decimals)
#A_UNMEM- labour union
#A_WKSTAT- full time status (==2 full time worker)

#H_IDNUM(20 digits)- Household number(in household data)
#PERIDNUM(22 digits) - Same number as household number(in personal data) + 2 digits
#They are to link household to a person to identify its state.


#take first 20 digits of PERIDNUM
raw_cps2024 <- raw_cps2024 %>%
  mutate(H_IDNUM = (substr(PERIDNUM, 1, 20)))

#connect two dataset by household number and select variables
cps_analysis <- raw_cps2024 %>%
  left_join(raw_hcps2024 %>% select(H_IDNUM, GESTFIPS), by = "H_IDNUM") %>%
  select(GESTFIPS, PERIDNUM, H_IDNUM, PEARNVAL, HRSWK, WKSWORK, A_WKSTAT, A_HGA, A_AGE, A_SEX, A_MARITL, PRDTRACE, PEHSPNON, A_HRSPAY, A_ERNLWT, A_UNMEM) %>%
  rename("state_id" = "GESTFIPS",
         "pp_id" = "PERIDNUM",
         "hh_id" = "H_IDNUM",
         "annual_income" = "PEARNVAL",
         "work_hpw" = "HRSWK",
         "work_wpy" = "WKSWORK",
         "education" = "A_HGA",
         "fulltime" = "A_WKSTAT",
         "age" = "A_AGE",
         "sex" = "A_SEX",
         "martial" = "A_MARITL",
         "race" = "PRDTRACE",
         "hispanic" = "PEHSPNON",
         "hwage" = "A_HRSPAY",
         "earn_wgt" = "A_ERNLWT",
         "lunion" = "A_UNMEM") %>%
  mutate(female = if_else(sex == 2, 1, 0),
         education = case_when(
           education == 31 ~ 0,
           education == 32 ~ 2,
           education == 33 ~ 6,
           education == 34 ~ 8,
           education == 35 ~ 9,
           education == 36 ~ 10,
           education == 37 ~ 11,
           education == 38 ~ 12,   # 12 but not graduated
           education == 39 ~ 12,
           education == 40 ~ 13,
           education %in% c(41, 42) ~ 14,
           education == 43 ~ 16,
           education == 44 ~ 18,
           education == 45 ~ 20,
           education == 46 ~ 21,
           TRUE ~ NA_real_
         ),
         experience = age - education - 6,
         experience2 = experience^2,
         earn_wgt = earn_wgt / 100,
         hispanic = if_else(hispanic == 1, 1, 0),
         afr_amer = if_else(race == 2, 1, 0),
         asian = if_else(race == 4, 1, 0),
         in_union = if_else(martial %in% c(1,2,3), 1, 0),
         lunion = if_else(lunion == 1, 1, 0),
         fulltime = if_else(fulltime == 2, 1, 0))


#hourly wage is A_HRSPAY if paied hourly, or annual income/ hour worked per year
cps_analysis <- cps_analysis %>%
  mutate(
    annual_hours = work_hpw * work_wpy,
    hwage = case_when(
      !is.na(hwage) & hwage > 0 ~ hwage / 100, #2 decimal places
      !is.na(annual_income) & !is.na(work_hpw) & !is.na(work_wpy) & work_hpw > 0 & work_wpy > 0 ~ annual_income / annual_hours,
      TRUE ~ NA_real_
    )
  )

#drop non fulltime workers (<35 hrs a week, <10 weeks a year),                                                                                                                                                                                                         drop missing values in hwage and education
cps_analysis <- cps_analysis %>%
  filter(fulltime == 1,
         earn_wgt > 0, #apply earn weight (drop the observation to 10,000)
         !is.na(hwage), 
         work_hpw >= 35, work_wpy >= 10, hwage < 100, hwage > 7.5,    #get rid of those whose hwage is <7.5 (Federal min wage) and > $100(outlier), worked less than 5 hrs per week/week per year
       !is.na(education), education > 0,
       experience > 0) %>% #get rid of negative experience           
  mutate(log_hwage = log(hwage),
         w = earn_wgt / mean(earn_wgt))

#combine state name with the id
#state_code <- read_csv("asecpub24csv/state_code.csv") |> zap_formats()

# cps_analysis <- cps_analysis %>%
#   left_join(state_code, by = "state_id")


######################
#visuals##############
######################

#Descriptive Statistics
descriptive <- cps_analysis %>%
  summarise(across(c(education, hwage, experience, female, experience, 
                     afr_amer, asian, hispanic, in_union, lunion),
                   list(mean = ~mean(., na.rm = TRUE),
                        min = ~min(., na.rm = TRUE),
                        max = ~max(., na.rm = TRUE),
                        obs = ~sum(!is.na(.)),
                        std_dev = ~sd(., na.rm = TRUE))
  ))

#scatter plot edu and log_hwage
cps_analysis %>%
  ggplot(aes(x = education, y = log_hwage)) +
  geom_point() +
  geom_smooth(method = "lm") +
  labs(x = "Education", y = "Log hourly wage") +
  theme_minimal()

#logwage distribution
cps_analysis %>%
  ggplot(aes(x = log_hwage)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  labs(x = "Log hourly wage") +
  theme_minimal()


#barplot by state
# cps_analysis %>%
#   group_by(state_name) %>%
#   summarise(mean = mean(hwage)) %>%
#   ggplot(aes(x = state_name, y = mean, fill = state_name)) +
#   geom_col(show.legend = F) +
#   labs(x = "State", y = "Hourly wage") +
#   theme(axis.text.x = element_text(angle = 90, hjust = 1))

# #boxplot by state
# cps_analysis %>%
#   ggplot(aes(x = state_name, y = hwage, fill = state_name)) +
#   geom_boxplot(show.legend = FALSE) +
#   labs(x = "State", y = "Hourly Wage") +
#   theme(axis.text.x = element_text(angle = 90, hjust = 1))

#indiana and corolado -outlier(get rid of outliers above)


######################
#OLS estimate#########
######################

#reference to DC
#cps_analysis$state_name <- relevel(factor(cps_analysis$state_name), ref = "District of Columbia")

######################
#base line estimates##

ols_base <- lm(log_hwage ~ education + experience + experience2, data = cps_analysis)
cat("---OLS baseline resilts---\n")
print(coeftest(ols_base, vcov = vcovHC(ols_base, type = "HC1")))
summary(ols_base)


#OLS estimates
ols_model <- lm(
  log_hwage ~ education + experience + experience2 + female +
    in_union + afr_amer + asian + hispanic + lunion,
  #weights = earn_wgt,
  data = cps_analysis #%>% filter(earn_wgt > 0) #get rid of wgt == 0
)
cat("--OLS results---\n")
print(coeftest(ols_model, vcov = vcovHC(ols_model, type = "HC1")))


#57,981 observation removed because of 0-weight. Obs is only 11392, but this properly represents the population officially.

#shoud I apply weights by cutting off majority of obs? considering DML connot include weights?


######################
#DML estimate#########
######################

#K(folds = 5), S(repetition) = 100

n_rep <- 100
estimates <- numeric(n_rep)

set.seed(123) #for reproductivity

for (i in 1:n_rep) {
  # Bootstrap sample: sample n rows with replacement
  data_boot <- cps_analysis[sample(1:nrow(cps_analysis), replace = TRUE), ]
  
  # Create DoubleMLData object with bootstrap sample
  dml_data <- DoubleMLData$new(
    data    = data.table(data_boot),
    y_col   = "log_hwage",
    d_cols  = "education",
    x_cols  = c("in_union", "female", "experience", "experience2",
                "afr_amer", "asian", "hispanic", "lunion")
  )
  
  learner <- lrn("regr.ranger")
  
  dml_model_cf <- DoubleMLPLR$new(
    data    = dml_data,
    ml_l    = learner,
    ml_m    = learner,
    n_folds = 5
  )
  
  dml_model_cf$fit()
  
  # Store the education coefficient
  estimates[i] <- dml_model_cf$coef["education"]
}



median_est <- median(estimates)
cat("Median DML Estimate for Education after 100 repetitions:", median_est, "\n")

summary(estimates)

sd(estimates)


############
#XG Boost###
############
library(DoubleML)
library(data.table)
library(mlr3)
library(mlr3learners) # XGBoostのため

n_rep <- 100
estimates <- numeric(n_rep)

set.seed(123) # 再現性のため

for (i in 1:n_rep) {
  # Bootstrapサンプル
  data_boot <- cps_analysis[sample(1:nrow(cps_analysis), replace = TRUE), ]
  
  dml_data <- DoubleMLData$new(
    data    = data.table(data_boot),
    y_col   = "log_hwage",
    d_cols  = "education",
    x_cols  = c("in_union", "female", "experience", "experience2",
                "afr_amer", "asian", "hispanic", "lunion")
  )
  
  # ここをXGBoostに
  learner <- lrn("regr.xgboost")
  
  dml_model_cf <- DoubleMLPLR$new(
    data    = dml_data,
    ml_l    = learner,
    ml_m    = learner,
    n_folds = 5
  )
  
  dml_model_cf$fit()
  
  estimates[i] <- dml_model_cf$coef["education"]
}

median_est <- median(estimates)
cat("Median DML Estimate for Education after 100 repetitions:", median_est, "\n")
summary(estimates)
sd(estimates)


############
#Lasso######
############
library(DoubleML)
library(data.table)
library(mlr3)
library(mlr3learners) # cv_glmnetが必要

# データ例: cps_analysis
n_rep <- 100
estimates <- numeric(n_rep)

set.seed(123) # 再現性

for (i in 1:n_rep) {
  # Bootstrapサンプル
  data_boot <- cps_analysis[sample(1:nrow(cps_analysis), replace = TRUE), ]
  
  dml_data <- DoubleMLData$new(
    data    = data.table(data_boot),
    y_col   = "log_hwage",
    d_cols  = "education",
    x_cols  = c("in_union", "female", "experience", "experience2",
                "afr_amer", "asian", "hispanic", "lunion")
  )
  
  # Lasso: alpha=1, cvでlambda選択
  learner <- lrn("regr.cv_glmnet", alpha = 1, nfolds= 5)
  
  dml_model_cf <- DoubleMLPLR$new(
    data    = dml_data,
    ml_l    = learner,
    ml_m    = learner,
    n_folds = 5
  )
  
  dml_model_cf$fit()
  
  estimates[i] <- dml_model_cf$coef["education"]
}

median_est <- median(estimates)
cat("Median DML Estimate for Education after 100 repetitions:", median_est, "\n")
summary(estimates)
sd(estimates)

############
#Generalized additive model (GAMs)
############
#manual implementation due to no GAM model in mlr3learners library (for auto DML)
library(mgcv)
set.seed(123)
n_rep <- 100
K <- 5
estimates <- numeric(n_rep)

for (i in 1:n_rep) {
  # ブートストラップサンプル
  idx_boot <- sample(1:nrow(cps_analysis), replace = TRUE)
  data_boot <- cps_analysis[idx_boot, ]
  
  n <- nrow(data_boot)
  folds <- sample(rep(1:K, length.out = n))
  res_y <- numeric(n)
  res_d <- numeric(n)
  
  for (k in 1:K) {
    train_idx <- which(folds != k)
    test_idx  <- which(folds == k)
    
    # y ~ controls (GAM)
    gam_y <- gam(log_hwage ~ s(experience) + s(experience2) + in_union + female +
                   afr_amer + asian + hispanic + lunion, 
                 data = data_boot[train_idx, ])
    pred_y <- predict(gam_y, newdata = data_boot[test_idx, ])
    res_y[test_idx] <- data_boot$log_hwage[test_idx] - pred_y
    
    # d ~ controls (GAM)
    gam_d <- gam(education ~ s(experience) + s(experience2) + in_union + female +
                   afr_amer + asian + hispanic + lunion, 
                 data = data_boot[train_idx, ])
    pred_d <- predict(gam_d, newdata = data_boot[test_idx, ])
    res_d[test_idx] <- data_boot$education[test_idx] - pred_d
  }
  
  # 残差回帰（回帰切片なし）
  dml_gam <- lm(res_y ~ res_d - 1)
  estimates[i] <- coef(dml_gam)[1]
}

# 推定値の中央値・標準偏差・要約
median_est <- median(estimates)
cat("Median DML-GAM Estimate for Education after 100 repetitions:", median_est, "\n")
summary(estimates)
sd(estimates)


############
#Neural nets
############
library(DoubleML)
library(data.table)
library(nnet)

n_rep <- 100
nnet_estimates <- numeric(n_rep)
K <- 5

set.seed(123)

for (i in 1:n_rep) {
  data_boot <- cps_analysis[sample(1:nrow(cps_analysis), replace = TRUE), ]
  n <- nrow(data_boot)
  
  folds <- sample(rep(1:K, length.out = n))
  nnet_res_y <- numeric(n)
  nnet_res_d <- numeric(n)
  
  for (k in 1:K) {
    train_idx <- which(folds != k)
    test_idx  <- which(folds == k)
    
    # スケーリングのためtrain/test分離
    train_data <- data_boot[train_idx, ]
    test_data  <- data_boot[test_idx, ]
    
    # log_hwage, education, experience, experience2のみスケーリング
    #scale test data with the info from train data.
    for (v in c("log_hwage", "education", "experience", "experience2")) {
      m <- mean(train_data[[v]])
      s <- sd(train_data[[v]])
      train_data[[v]] <- (train_data[[v]] - m) / s
      test_data[[v]]  <- (test_data[[v]]  - m) / s
    }
    
    # nnet: y ~ controls
    nn_y <- nnet(log_hwage ~ experience + experience2 + in_union + female +
                   afr_amer + asian + hispanic + lunion,
                 data = train_data, linout = TRUE, size = 5, maxit = 500, trace = FALSE)
    nn_pred_y <- predict(nn_y, newdata = test_data)
    nnet_res_y[test_idx] <- test_data$log_hwage - nn_pred_y
    
    # nnet: d ~ controls
    nn_d <- nnet(education ~ experience + experience2 + in_union + female +
                   afr_amer + asian + hispanic + lunion,
                 data = train_data, linout = TRUE, size = 5, maxit = 500, trace = FALSE)
    nn_pred_d <- predict(nn_d, newdata = test_data)
    nnet_res_d[test_idx] <- test_data$education - nn_pred_d
  }
  
  # 残差回帰（切片なし）
  nnet_dml <- lm(nnet_res_y ~ nnet_res_d - 1)
  nnet_estimates[i] <- coef(nnet_dml)[1]
}


median_est <- median(nnet_estimates)
cat("Median DML-NNet Estimate for Education after 100 repetitions:", median_est, "\n")
summary(nnet_estimates)
sd(nnet_estimates)

#unscale the coefficients
sd_edu <- sd(cps_analysis$education)
sd_wage <- sd(cps_analysis$log_hwage)
unscaled_est <- median_est * (sd_wage / sd_edu)
unscaled_sd <- sd(nnet_estimates) * (sd_wage / sd_edu)
unscaled_est
unscaled_sd


############
#z-score
############
beta_ols <- 8.819e-02
beta_dml <- 0.0874
se_ols <- 8.189e-04
se_dml <- 0.00075

# 差のz統計量を計算
z_value <- (beta_ols - beta_dml) / sqrt(se_ols^2 + se_dml^2)

# p値を計算（両側検定）
p_value <- 2 * (1 - pnorm(abs(z_value)))

# output
cat("=== difference test ===\n")
cat("z-value:", round(z_value, 3), "\n")
cat("p-value:", round(p_value, 4), "\n")

if (p_value < 0.05) {
  cat("→ significant difference at 5%\n")
} else {
  cat("→ No significant difference at 5%\n")
}

############
#simulation#
############

#simulation for a mincerian equation

library(DoubleML)
library(mlr3)
library(mlr3learners)
library(data.table)
library(mgcv)
library(nnet)

set.seed(123)

# ======== simulation settings ========
n_sim <- 100
n <- 1000
true_ate <- 0.1
K <- 5

#results box
ols_estimates <- numeric(n_sim)
rf_estimates <- numeric(n_sim)
xg_estimates <- numeric(n_sim)
lasso_estimates <- numeric(n_sim)
gam_estimates <- numeric(n_sim)
nnet_estimates <- numeric(n_sim)


for (i in 1:n_sim) {
  # ======== create sythesized data ========
  education <- sample(8:20, n, replace = TRUE)
  experience <- pmax(0, rnorm(n, 15, 5))
  experience2 <- experience^2
  
  # true model: the effect of education is 0.1
  wage <- 0.1 * education +
    0.03 * experience -
    0.0005 * experience2 +
    rnorm(n)
  
  sim_data <- data.frame(wage, education, experience, experience2)
  
  # ======== OLS estimate ========
  ols_model <- lm(wage ~ education + experience + experience2, data = sim_data)
  ols_estimates[i] <- coef(ols_model)["education"]
  
  # ======== DML estimate (rf, xg, lasso, gams, nnet) ========
  dml_data <- DoubleMLData$new(
    data = data.table(sim_data),
    y_col = "wage",
    d_cols = "education",
    x_cols = c("experience", "experience2")
  )
  #Random forest
  rf <- lrn("regr.ranger")
  rf_model <- DoubleMLPLR$new(data = dml_data, ml_l = rf, ml_m = rf, n_folds = K)
  rf_model$fit()
  rf_estimates[i] <- rf_model$coef
  
  #Boosted trees
  xg <- lrn("regr.xgboost")
  xg_model <- DoubleMLPLR$new(data = dml_data, ml_l = xg, ml_m = xg, n_folds = K)
  xg_model$fit()
  xg_estimates[i] <- xg_model$coef
  
  #lasso
  lasso <- lrn("regr.cv_glmnet", alpha = 1)
  lasso_model <- DoubleMLPLR$new(data = dml_data, ml_l = lasso, ml_m = lasso, n_folds = K)
  lasso_model$fit()
  lasso_estimates[i] <- lasso_model$coef

  

  #GAMs
  # ===== クロスフィッティング用のfold分割 =====
  folds <- sample(rep(1:K, length.out = n))
  gam_res_y <- numeric(n)
  gam_res_d <- numeric(n)
  
  for (k in 1:K) {
    train_idx <- which(folds != k)
    test_idx <- which(folds == k)
    
    # y ~ controls をGAMで
    gam_y <- gam(wage ~ s(experience) + s(experience2), data = sim_data[train_idx, ])
    gam_pred_y <- predict(gam_y, newdata = sim_data[test_idx, ])
    gam_res_y[test_idx] <- sim_data$wage[test_idx] - gam_pred_y
    
    # d ~ controls をGAMで
    gam_d <- gam(education ~ s(experience) + s(experience2), data = sim_data[train_idx, ])
    gam_pred_d <- predict(gam_d, newdata = sim_data[test_idx, ])
    gam_res_d[test_idx] <- sim_data$education[test_idx] - gam_pred_d
  }
  
  # ===== 残差回帰（切片なし） =====
  gam_dml <- lm(gam_res_y ~ gam_res_d - 1)
  gam_estimates[i] <- coef(gam_dml)[1]
  
  
  #neural networks
  # neural networks
  folds <- sample(rep(1:K, length.out = n))
  nnet_res_y <- numeric(n)
  nnet_res_d <- numeric(n)
  
  for (k in 1:K) {
    train_idx <- which(folds != k)
    test_idx <- which(folds == k)
    
    # スケーリング用データの準備
    train_data <- sim_data[train_idx, ]
    test_data  <- sim_data[test_idx, ]
    
    # 各変数について、trainのmean/sdでスケール
    for (v in c("wage", "education", "experience", "experience2")) {
      m <- mean(train_data[[v]])
      s <- sd(train_data[[v]])
      train_data[[v]] <- (train_data[[v]] - m) / s
      test_data[[v]]  <- (test_data[[v]]  - m) / s
    }
    
    # y ~ controls
    nn_y <- nnet(wage ~ experience + experience2, data = train_data, linout = TRUE, size = 5, maxit = 500, trace = FALSE)
    nn_pred_y <- predict(nn_y, newdata = test_data)
    nnet_res_y[test_idx] <- test_data$wage - nn_pred_y
    
    # d ~ controls
    nn_d <- nnet(education ~ experience + experience2, data = train_data, linout = TRUE, size = 5, maxit = 500, trace = FALSE)
    nn_pred_d <- predict(nn_d, newdata = test_data)
    nnet_res_d[test_idx] <- test_data$education - nn_pred_d
  
  # 残差回帰
  nnet_dml <- lm(nnet_res_y ~ nnet_res_d - 1)
  nnet_estimates[i] <- coef(nnet_dml)[1]
  
  }
}

# ======== summary of results ========
summary_results <- data.frame(
  Model = c("OLS", "Random Forest", "XG Boost", "Lasso", "GAMs", "Neural Nets"),
  Mean = c(mean(ols_estimates), mean(rf_estimates), mean(xg_estimates), mean(lasso_estimates), mean(gam_estimates), mean(nnet_estimates)),
  SD = c(sd(ols_estimates), sd(rf_estimates), sd(xg_estimates), sd(lasso_estimates), sd(gam_estimates), sd(nnet_estimates)),
  Bias = c(mean(ols_estimates) - true_ate, 
           mean(rf_estimates) - true_ate, 
           mean(xg_estimates) - true_ate, 
           mean(lasso_estimates) - true_ate, 
           mean(gam_estimates) - true_ate, 
           mean(nnet_estimates) - true_ate),
  RMSE = c(
    sqrt(mean((ols_estimates - true_ate)^2)),
    sqrt(mean((rf_estimates - true_ate)^2)),
    sqrt(mean((xg_estimates - true_ate)^2)),
    sqrt(mean((lasso_estimates - true_ate)^2)),
    sqrt(mean((gam_estimates - true_ate)^2)),
    sqrt(mean((nnet_estimates - true_ate)^2))
  )
)

print(summary_results)


#next: with the simulated cps data (n = 49019 and the same variables)

library(DoubleML)
library(mlr3)
library(mlr3learners)
library(data.table)
library(mgcv)
library(nnet)

set.seed(123)

# ======== simulation settings ========
n_sim <- 100
n <- 49019
true_ate <- 0.1
K <- 5

#results box
ols_estimates <- numeric(n_sim)
rf_estimates <- numeric(n_sim)
xg_estimates <- numeric(n_sim)
lasso_estimates <- numeric(n_sim)
gam_estimates <- numeric(n_sim)
nnet_estimates <- numeric(n_sim)


for (i in 1:n_sim) {
  # ======== create sythesized data ========
  education <- sample(8:20, n, replace = TRUE)
  experience <- pmax(0, rnorm(n, 23, 13))
  experience2 <- experience^2
  female <- rbinom(n, 1, 0.45)
  afr_amer <- rbinom(n, 1, 0.11)
  asian <- rbinom(n, 1, 0.07)
  hispanic <- rbinom(n, 1, 0.2)
  in_union <- rbinom(n, 1, 0.6)
  lunion <- rbinom(n, 1, 0.02)
  
  # true model: the effect of education is 0.1
  wage <- 0.1 * education +
    0.03 * experience -
    0.0005 * experience2 -
    0.15 * female -
    0.05 * afr_amer +
    0.03 * asian +
    0.02 * hispanic +
    0.07 * in_union +
    0.1 * lunion +
    rnorm(n)
  
  sim_data <- data.frame(wage, education, experience, experience2, female, afr_amer, asian, hispanic, in_union, lunion)
  
  # ======== OLS estimate ========
  ols_model <- lm(wage ~ education + experience + experience2, data = sim_data)
  ols_estimates[i] <- coef(ols_model)["education"]
  
  # ======== DML estimate (rf, xg, lasso, gams, nnet) ========
  dml_data <- DoubleMLData$new(
    data = data.table(sim_data),
    y_col = "wage",
    d_cols = "education",
    x_cols = c("experience", "experience2")
  )
  #Random forest
  rf <- lrn("regr.ranger")
  rf_model <- DoubleMLPLR$new(data = dml_data, ml_l = rf, ml_m = rf, n_folds = K)
  rf_model$fit()
  rf_estimates[i] <- rf_model$coef
  
  #Boosted trees
  xg <- lrn("regr.xgboost")
  xg_model <- DoubleMLPLR$new(data = dml_data, ml_l = xg, ml_m = xg, n_folds = K)
  xg_model$fit()
  xg_estimates[i] <- xg_model$coef
  
  #lasso
  lasso <- lrn("regr.cv_glmnet", alpha = 1)
  lasso_model <- DoubleMLPLR$new(data = dml_data, ml_l = lasso, ml_m = lasso, n_folds = K)
  lasso_model$fit()
  lasso_estimates[i] <- lasso_model$coef
  
  
  
  #GAMs
  # ===== クロスフィッティング用のfold分割 =====
  folds <- sample(rep(1:K, length.out = n))
  gam_res_y <- numeric(n)
  gam_res_d <- numeric(n)
  
  for (k in 1:K) {
    train_idx <- which(folds != k)
    test_idx <- which(folds == k)
    
    # y ~ controls をGAMで
    gam_y <- gam(wage ~ s(experience) + s(experience2), data = sim_data[train_idx, ])
    gam_pred_y <- predict(gam_y, newdata = sim_data[test_idx, ])
    gam_res_y[test_idx] <- sim_data$wage[test_idx] - gam_pred_y
    
    # d ~ controls をGAMで
    gam_d <- gam(education ~ s(experience) + s(experience2), data = sim_data[train_idx, ])
    gam_pred_d <- predict(gam_d, newdata = sim_data[test_idx, ])
    gam_res_d[test_idx] <- sim_data$education[test_idx] - gam_pred_d
  }
  
  # ===== 残差回帰（切片なし） =====
  gam_dml <- lm(gam_res_y ~ gam_res_d - 1)
  gam_estimates[i] <- coef(gam_dml)[1]
  
  
  #neural networks
  # neural networks
  folds <- sample(rep(1:K, length.out = n))
  nnet_res_y <- numeric(n)
  nnet_res_d <- numeric(n)
  
  for (k in 1:K) {
    train_idx <- which(folds != k)
    test_idx <- which(folds == k)
    
    # スケーリング用データの準備
    train_data <- sim_data[train_idx, ]
    test_data  <- sim_data[test_idx, ]
    
    # 各変数について、trainのmean/sdでスケール
    for (v in c("wage", "education", "experience", "experience2")) {
      m <- mean(train_data[[v]])
      s <- sd(train_data[[v]])
      train_data[[v]] <- (train_data[[v]] - m) / s
      test_data[[v]]  <- (test_data[[v]]  - m) / s
    }
    
    # y ~ controls
    nn_y <- nnet(wage ~ experience + experience2, data = train_data, linout = TRUE, size = 5, maxit = 500, trace = FALSE)
    nn_pred_y <- predict(nn_y, newdata = test_data)
    nnet_res_y[test_idx] <- test_data$wage - nn_pred_y
    
    # d ~ controls
    nn_d <- nnet(education ~ experience + experience2, data = train_data, linout = TRUE, size = 5, maxit = 500, trace = FALSE)
    nn_pred_d <- predict(nn_d, newdata = test_data)
    nnet_res_d[test_idx] <- test_data$education - nn_pred_d
    
    # 残差回帰
    nnet_dml <- lm(nnet_res_y ~ nnet_res_d - 1)
    nnet_estimates[i] <- coef(nnet_dml)[1]
    
  }
}

# ======== summary of results ========
summary_results <- data.frame(
  Model = c("OLS", "Random Forest", "XG Boost", "Lasso", "GAMs", "Neural Nets"),
  Mean = c(mean(ols_estimates), mean(rf_estimates), mean(xg_estimates), mean(lasso_estimates), mean(gam_estimates), mean(nnet_estimates)),
  SD = c(sd(ols_estimates), sd(rf_estimates), sd(xg_estimates), sd(lasso_estimates), sd(gam_estimates), sd(nnet_estimates)),
  Bias = c(mean(ols_estimates) - true_ate, 
           mean(rf_estimates) - true_ate, 
           mean(xg_estimates) - true_ate, 
           mean(lasso_estimates) - true_ate, 
           mean(gam_estimates) - true_ate, 
           mean(nnet_estimates) - true_ate),
  RMSE = c(
    sqrt(mean((ols_estimates - true_ate)^2)),
    sqrt(mean((rf_estimates - true_ate)^2)),
    sqrt(mean((xg_estimates - true_ate)^2)),
    sqrt(mean((lasso_estimates - true_ate)^2)),
    sqrt(mean((gam_estimates - true_ate)^2)),
    sqrt(mean((nnet_estimates - true_ate)^2))
  )
)

print(summary_results)

#############
#box plot####
#############
library(tidyr)
library(ggplot2)
results_df <- data.frame(
  OLS = ols_estimates,
  RandomForest = rf_estimates,
  XGBoost = xg_estimates,
  Lasso = lasso_estimates,
  GAMs = gam_estimates,
  NeuralNets = nnet_estimates
)
results_long <- pivot_longer(results_df, cols = everything(), names_to = "Model", values_to = "Estimate")
ggplot(results_long, aes(x = Model, y = Estimate, fill = Model)) +
  geom_boxplot(alpha = 0.7) +
  geom_hline(yintercept = 0.1, linetype = "dashed", color = "black") +
  theme_minimal() +
  labs(title = "Distribution of Estimated ATE by Model",
       y = "Estimated ATE", x = "Model") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


