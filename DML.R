#This script includes only running models part (run 'dml_paper' first to get and clean the dataset.)
#run every model in one bootstrap to get a variance-covariance matrix.
#S = 100 (repetition) is applied only for the first bootstrap for coomputational limitation.

## ========================
## summary of this script
## ========================
#cps_analysis is created in data_prep script and ready
# Conduct DML and OLS algorithm and compare the differences. 
# OLS with 4 kinds: baseline (mincer), all vars, with edu2, and with college dummy.
# DML has 5 ML algorithms working inside: RF, XGB, Lasso, GAMs, NN.
# RF, XGB, NN's parameters are tuned by 3-folds nested CV (works inside cross-fitting 4 training folds)
# t-stat of the differences (specifically, var/cov matrix) are calculated by 500 times bootstrapping the data.
# For t-stat, the estimates (numerator) are point est ones (with repitition=100).
# Pararele processing using 8 core - 1
# est. running time is 20 hours



## =============================
## Setup
## =============================
library(data.table)
library(DoubleML)
library(mlr3)


library(mlr3learners)  # ranger / xgboost / glmnet
library(mgcv)          # GAM (hand-made partialling-out)
library(nnet)          # NN  (hand-made partialling-out)
library(lmtest)
library(sandwich)
library(mlr3tuning)
library(paradox)
library(future)
library(future.apply)

## ============================
## Pararel Processing settings
## ============================


# Show available cores
cat("Available cores:", availableCores(), "\n")

# activate pararel processing (use all cores - 1)
n_workers <- max(1, availableCores() - 1)
cat("Using", n_workers, "workers\n")
plan(multisession, workers = n_workers)

# acrtivate in mlr3
options(mlr3.exec_future = TRUE)

# Show progress
library(progressr)
handlers(global = TRUE)


## ----------
set.seed(123)

# Controls
x_cols <- c("in_union","female","experience","experience2",
            "afr_amer","asian","hispanic","lunion")

K <- 5          # cross-fitting folds
B <- 100        # bootstrap replications for SE/VCOV（can be 300〜1000）
R <- 100        # repeats for point estimates on full data（report the median）
# R <- 10 # for testing

n <- nrow(cps_analysis)

## =============================
## Helpers
## =============================

# Make K-fold split id (length n, values in 1..K)
make_fold_id <- function(n, K) sample(rep(1:K, length.out = n))

#cross-validation helper
make_autotuner <- function(learner, search_space,
                           inner_folds = 3, #nest-CV (n. of folds inside traning data) is default
                           n_evals = 30, #30 random picks is defalt
                           measure = msr("regr.mse"),
                           tuner = tnr("random_search")) #randomly choose between values
                           {
  AutoTuner$new(
    learner = learner,
    resampling = rsmp("cv", folds = inner_folds),
    measure = measure,
    search_space = search_space,
    terminator = trm("evals", n_evals = n_evals),
    tuner = tuner,
    store_models = FALSE
  )
}

#RF CV tuner
make_rf_autotuner <- function() {
  base <- lrn("regr.ranger",
              num.trees = 500,
              respect.unordered.factors = "order",
              num.threads = 1)
  ps <- ps(
    mtry.ratio      = p_dbl(0.2, 1.0), #continuous values between them is considered
    min.node.size   = p_int(2, 20),
    sample.fraction = p_dbl(0.5, 1.0)
  )
  make_autotuner(base, ps, inner_folds = 3, n_evals = 30) #30 times of random search between 2 values.
}

#XGB auto tuner
make_xgb_autotuner <- function() {
  base <- lrn("regr.xgboost",
              objective = "reg:squarederror",
              nthread = 1)
  ps <- ps(
    eta               = p_dbl(0.05, 0.3),
    max_depth         = p_int(2, 8),
    subsample         = p_dbl(0.5, 1.0),
    colsample_bytree  = p_dbl(0.5, 1.0),
    nrounds           = p_int(100, 600)
  )
  make_autotuner(base, ps, inner_folds = 3, n_evals = 30)
}

#lasso (if you want elastic net)
# make_glmnet_autotuner <- function() {
#   base <- lrn("regr.cv_glmnet", nfolds = 5, standardize = TRUE)
#   ps <- ps(
#     alpha = p_dbl(0, 1)  # 0=Ridge, 1=Lasso
#   )
#   make_autotuner(base, ps, inner_folds = 3, n_evals = 20)
# }

#NN auto tuner
make_nnet_autotuner <- function() {
  base <- lrn("regr.nnet",
              skip = TRUE, maxit = 500, trace = FALSE, MaxNWts = 5000)
  ps <- ps(
    size  = p_int(2, 10),
    decay = p_dbl(1e-4, 1e-1, logscale = TRUE)
  )
  make_autotuner(base, ps, inner_folds = 3, n_evals = 30)
}



dml_manual_weighted <- function(df, x_cols, K, fold_id, learner_y, learner_d) {
  stopifnot(all(c("log_hwage","education","w") %in% names(df)))
  df <- data.table::as.data.table(df)

  n <- nrow(df)
  res_y <- numeric(n)
  res_d <- numeric(n)

  for (k in 1:K) {
    tr <- which(fold_id != k); te <- which(fold_id == k)

    # ---- g(X): E[Y|X]
    task_y <- TaskRegr$new(
      id = paste0("y_", k),
      backend = df[tr, c("log_hwage", x_cols, "w"), with = FALSE],
      target = "log_hwage"
    )
    task_y$set_col_roles("w", roles = "weights_learner")  # apply weight
    learner_y$train(task_y)
    pred_y <- learner_y$predict_newdata(newdata = df[te, x_cols, with = FALSE])$response
    res_y[te] <- df$log_hwage[te] - pred_y

    # ---- m(X): E[D|X]
    task_d <- TaskRegr$new(
      id = paste0("d_", k),
      backend = df[tr, c("education", x_cols, "w"), with = FALSE],
      target = "education"
    )
    task_d$set_col_roles("w", roles = "weights_learner")  # apply weight
    learner_d$train(task_d)
    pred_d <- learner_d$predict_newdata(newdata = df[te, x_cols, with = FALSE])$response
    res_d[te] <- df$education[te] - pred_d
  }

  reg <- lm(res_y ~ res_d - 1, weights = df$w)
  vc  <- sandwich::vcovHC(reg, type = "HC1")
  se  <- sqrt(diag(vc))[["res_d"]]
  est <- coef(reg)[["res_d"]]
  list(est = est, se = se)
}



# ===========================================
# 1) Random Forest (ranger)
# ===========================================
fit_dml_rf_w <- function(df, x_cols, K, fold_id = NULL) {
  if (is.null(fold_id)) fold_id <- make_fold_id(nrow(df), K)
  # adjust parameters
  lrn_rf_y <- make_rf_autotuner()
  lrn_rf_d <- make_rf_autotuner()

  dml_manual_weighted(df, x_cols, K, fold_id, lrn_rf_y, lrn_rf_d)
}

# ===========================================
# 2) XGBoost
# ===========================================
fit_dml_xgb_w <- function(df, x_cols, K, fold_id = NULL) {
  if (is.null(fold_id)) fold_id <- make_fold_id(nrow(df), K)
  # parameters
  lrn_xgb_y <- make_xgb_autotuner()
  lrn_xgb_d <- make_xgb_autotuner()

  dml_manual_weighted(df, x_cols, K, fold_id, lrn_xgb_y, lrn_xgb_d)
}

# ===========================================
# 3) Lasso（cv_glmnet）
# ===========================================
fit_dml_lasso_w <- function(df, x_cols, K, fold_id = NULL) {
  if (is.null(fold_id)) fold_id <- make_fold_id(nrow(df), K)
  # alpha=1 is lasso
  lrn_lasso_y <- lrn("regr.cv_glmnet", alpha = 1, nfolds = 5, standardize = TRUE)
  lrn_lasso_d <- lrn_lasso_y$clone(deep = TRUE)

  dml_manual_weighted(df, x_cols, K, fold_id, lrn_lasso_y, lrn_lasso_d)
}

# ==========manual DML=========== 

#########
# Partialling-out with GAM (same fold split as DML)
#########

fit_dml_gam <- function(df, x_cols, K, fold_id){
  nb <- nrow(df)
  res_y <- numeric(nb)
  res_d <- numeric(nb)
  for(k in 1:K){
    tr <- which(fold_id != k); te <- which(fold_id == k)
    # y ~ controls (smooth on cont.; linear on dummies)
    f_y <- as.formula(
      "log_hwage ~ s(experience) + in_union + female + afr_amer + asian + hispanic + lunion"
    ) #experience2 is removed for multicollinearity
    gy <- mgcv::gam(f_y, data = df[tr, ], weights = df$w[tr], method = 'REML')
    pred_y <- predict(gy, newdata = df[te, ])
    res_y[te] <- df$log_hwage[te] - pred_y

    # d ~ controls
    f_d <- as.formula(
      "education ~ s(experience) + in_union + female + afr_amer + asian + hispanic + lunion"
    ) #experience2 is removed for multicollinearity
    gd <- mgcv::gam(f_d, data = df[tr, ], weights = df$w[tr], method = 'REML')
    pred_d <- predict(gd, newdata = df[te, ])
    res_d[te] <- df$education[te] - pred_d
  }
  reg <- lm(res_y ~ res_d - 1, weights = df$w)
  vc  <- sandwich::vcovHC(reg, type = "HC1")
  se  <- sqrt(diag(vc))[["res_d"]]
  est <- coef(reg)[["res_d"]]
  return(list(est = est, se = se))

}

#########
# NN with inner CV (size, decay) — weights via roles
#########
fit_dml_nn <- function(df, x_cols, K, fold_id){
  stopifnot(all(c("log_hwage","education","w", x_cols) %in% names(df)))
  nb <- nrow(df)
  res_y <- numeric(nb)
  res_d <- numeric(nb)

  for(k in seq_len(K)){
    tr <- which(fold_id != k); te <- which(fold_id == k)
    train <- df[tr, ]
    test  <- df[te, ]

    # per-fold scaling for continuous X (do NOT scale targets)
    for(v in c("experience","experience2")){
      m <- mean(train[[v]]); s <- sd(train[[v]]); if(is.na(s) || s == 0) s <- 1
      train[[v]] <- (train[[v]] - m) / s
      test[[v]]  <- (test[[v]]  - m) / s
    }

    # ---- Task for g(X): E[Y|X] ----
    dat_y <- train[, c("log_hwage", x_cols, "w"), drop = FALSE]
    task_y <- TaskRegr$new(id = paste0("nn_y_", k), backend = dat_y, target = "log_hwage")
    task_y$set_col_roles("w", roles = "weights_learner")

    # ---- Task for m(X): E[D|X] ----
    dat_d <- train[, c("education", x_cols, "w"), drop = FALSE]
    task_d <- TaskRegr$new(id = paste0("nn_d_", k), backend = dat_d, target = "education")
    task_d$set_col_roles("w", roles = "weights_learner")

    # AutoTuners
    at_y <- make_nnet_autotuner(); at_y$train(task_y)
    at_d <- make_nnet_autotuner(); at_d$train(task_d)

    # Predict on test X only (no w column)
    pred_y <- at_y$predict_newdata(test[, x_cols, drop = FALSE])$response
    pred_d <- at_d$predict_newdata(test[, x_cols, drop = FALSE])$response

    res_y[te] <- test$log_hwage - pred_y
    res_d[te] <- test$education - pred_d
  }

  reg <- lm(res_y ~ res_d - 1, weights = df$w)
  vc  <- sandwich::vcovHC(reg, type = "HC1")
  list(est = coef(reg)[["res_d"]], se = sqrt(diag(vc))[["res_d"]])
}


## =============================
## (A) point estimates (R=100)
## =============================

#est df
point_draws <- data.frame(
  ols = numeric(R),
  ols_base = numeric(R),
  ols_edu2 = numeric(R),
  ols_college = numeric(R),
  dml_rf = numeric(R),
  dml_xgb = numeric(R),
  dml_lasso = numeric(R),
  dml_gam = numeric(R),
  dml_nn = numeric(R)
)
 #se df
point_ses <- point_draws

for(r in 1:R){
  fold_id <- make_fold_id(n, K)

  #OLS-baseline
  ols_base_fit <- lm(
    log_hwage ~ education + experience + experience2, data= cps_analysis, weights = w
  )
  point_draws$ols_base[r] <- coef(ols_base_fit)[["education"]]
  vc_ols_base = sandwich::vcovHC(ols_base_fit, type = 'HC1')
  point_ses$ols_base[r] <- sqrt(diag(vc_ols_base))[['education']]
  # OLS
  ols_fit <- lm(
    log_hwage ~ education + experience + experience2 + female +
      in_union + afr_amer + asian + hispanic + lunion,
    data = cps_analysis, weights = w
  )
  point_draws$ols[r] <- coef(ols_fit)[["education"]]
  vc = sandwich::vcovHC(ols_fit, type = 'HC1')
  point_ses$ols[r] <- sqrt(diag(vc))[['education']]

  #OLS with education2
    ols_edu2 <- lm(
    log_hwage ~ education + education2 + experience + experience2 + female +
      in_union + afr_amer + asian + hispanic + lunion,
    data = cps_analysis, weights = w
  )
  point_draws$olsedu2[r] <- coef(ols_edu2)[["education"]]
  vc = sandwich::vcovHC(ols_edu2, type = 'HC1')
  point_ses$olsedu2[r] <- sqrt(diag(vc))[['education']]

  #OLS with college dummy
    ols_college <- lm(
    log_hwage ~ college + experience + experience2 + female +
      in_union + afr_amer + asian + hispanic + lunion,
    data = cps_analysis, weights = w
  )
  point_draws$olscollege[r] <- coef(ols_college)[["college"]]
  vc = sandwich::vcovHC(ols_college, type = 'HC1')
  point_ses$olscollege[r] <- sqrt(diag(vc))[['college']]

  # DML variants with the same fold
  res_rf    <- fit_dml_rf_w(cps_analysis, x_cols, K, fold_id)
  res_xgb  <- fit_dml_xgb_w(cps_analysis, x_cols, K, fold_id)
  res_lasso <- fit_dml_lasso_w(cps_analysis, x_cols, K, fold_id)
  res_gam   <- fit_dml_gam(cps_analysis, x_cols, K, fold_id)
  res_nn    <- fit_dml_nn(cps_analysis, x_cols, K, fold_id)

  point_draws$dml_rf[r] <- res_rf$est
  point_draws$dml_xgb[r] <- res_xgb$est
  point_draws$dml_lasso[r] <- res_lasso$est
  point_draws$dml_gam[r] <- res_gam$est
  point_draws$dml_nn[r] <- res_nn$est

  point_ses$dml_rf[r] <- res_rf$se
  point_ses$dml_xgb[r] <- res_xgb$se
  point_ses$dml_lasso[r] <- res_lasso$se
  point_ses$dml_gam[r] <- res_gam$se
  point_ses$dml_nn[r] <- res_nn$se


  if(r %% 5 == 0){
    message(paste0("Iteration #:", r, "for point est."))
  }
}

summary_tab <- data.frame(
  model = colnames(point_draws),
  median_est = apply(point_draws, 2, median, na.rm = TRUE),
  median_se  = apply(point_ses, 2, median, na.rm = TRUE)
)
point_est <- apply(point_draws, 2, median, na.rm = TRUE)
print(summary_tab)


## =============================
## (B) Bootstrap: each boostrap, report Var and Cov
## =============================

# ---- initialize matrix ----
M <- 8  # models excluding OLS_base
est_mat <- matrix(NA_real_, nrow = B, ncol = M)
colnames(est_mat) <- c("ols","ols_edu2","ols_coll", "dml_rf","dml_xgb","dml_lasso","dml_gam","dml_nn")

for(b in 1:B){
  idx_b <- sample.int(n, size = n, replace = TRUE)
  db    <- cps_analysis[idx_b, ]
  nb    <- nrow(db)
  
  # sample-specific weights if needed
  db$w <- db$w  # make explicit in bootstrap

  fold_id <- make_fold_id(nb, K)

  # ---- OLS ----
  ols_b <- coef(lm(
    log_hwage ~ education + experience + experience2 + female +
      in_union + afr_amer + asian + hispanic + lunion,
    data = db, weights = db$w
  ))[["education"]]

  # # ---- OLS with edu2 ----
    ols_e2 <- coef(lm(
    log_hwage ~ education + education2 + experience + experience2 + female +
      in_union + afr_amer + asian + hispanic + lunion,
    data = db, weights = db$w
  ))[["education"]]

  # # ---- OLS with college dummy ----
    ols_coll <- coef(lm(
    log_hwage ~ college + experience + experience2 + female +
      in_union + afr_amer + asian + hispanic + lunion,
    data = db, weights = db$w
  ))[["college"]]

  # ---- DMLs ----
  rf_b    <- fit_dml_rf_w(db, x_cols, K, fold_id)$est
  xgb_b   <- fit_dml_xgb_w(db, x_cols, K, fold_id)$est
  lasso_b <- fit_dml_lasso_w(db, x_cols, K, fold_id)$est
  gam_b   <- fit_dml_gam(db, x_cols, K, fold_id)$est
  nn_b    <- fit_dml_nn(db, x_cols, K, fold_id)$est

  est_mat[b, ] <- c(ols_b, ols_e2, ols_coll, rf_b, xgb_b, lasso_b, gam_b, nn_b)

  if(b %% 10 == 0){
    message(paste0("Bootstrap iteration #", b))
  }
}

# ---- Bootstrap variance-covariance ----
V <- cov(est_mat, use = "pairwise.complete.obs")
print(round(V, 5))

## in case you skip (A)
point_est <- c(
  ols       = 0.087,
  ols_edu2  = 0.031,
  ols_coll  = 0.45,
  dml_rf    = 0.0856,
  dml_xgb   = 0.0856,
  dml_lasso = 0.0859,
  dml_gam   = 0.0871,
  dml_nn    = 0.085
)



# ---- SE of differences: OLS - DML ----
se_diff_ols_rf     <- sqrt(V["ols","ols"] + V["dml_rf","dml_rf"] - 2*V["ols","dml_rf"])
se_diff_ols_xg     <- sqrt(V["ols","ols"] + V["dml_xgb","dml_xgb"] - 2*V["ols","dml_xgb"])
se_diff_ols_lasso  <- sqrt(V["ols","ols"] + V["dml_lasso","dml_lasso"] - 2*V["ols","dml_lasso"])
se_diff_ols_gam    <- sqrt(V["ols","ols"] + V["dml_gam","dml_gam"] - 2*V["ols","dml_gam"])
se_diff_ols_nn     <- sqrt(V["ols","ols"] + V["dml_nn","dml_nn"] - 2*V["ols","dml_nn"])

# ---- t-stats using point_est from (A) ----
t_ols_rf     <- (point_est["ols"] - point_est["dml_rf"])    / se_diff_ols_rf
t_ols_xg     <- (point_est["ols"] - point_est["dml_xgb"])   / se_diff_ols_xg
t_ols_lasso  <- (point_est["ols"] - point_est["dml_lasso"]) / se_diff_ols_lasso
t_ols_gam    <- (point_est["ols"] - point_est["dml_gam"])   / se_diff_ols_gam
t_ols_nn     <- (point_est["ols"] - point_est["dml_nn"])    / se_diff_ols_nn

# ---- two-sided p-values ----
p_ols_rf     <- 2 * pnorm(-abs(t_ols_rf))
p_ols_xg     <- 2 * pnorm(-abs(t_ols_xg))
p_ols_lasso  <- 2 * pnorm(-abs(t_ols_lasso))
p_ols_gam    <- 2 * pnorm(-abs(t_ols_gam))
p_ols_nn     <- 2 * pnorm(-abs(t_ols_nn))

## ======================
## edu2 vs DML
## ======================
# ---- SE of differences: OLS_E2 - DML ----
se_diff_ols_edu2_rf     <- sqrt(V["ols_edu2","ols_edu2"] + V["dml_rf","dml_rf"] - 2*V["ols_edu2","dml_rf"])
se_diff_ols_edu2_xg     <- sqrt(V["ols_edu2","ols_edu2"] + V["dml_xgb","dml_xgb"] - 2*V["ols_edu2","dml_xgb"])
se_diff_ols_edu2_lasso  <- sqrt(V["ols_edu2","ols_edu2"] + V["dml_lasso","dml_lasso"] - 2*V["ols_edu2","dml_lasso"])
se_diff_ols_edu2_gam    <- sqrt(V["ols_edu2","ols_edu2"] + V["dml_gam","dml_gam"] - 2*V["ols_edu2","dml_gam"])
se_diff_ols_edu2_nn     <- sqrt(V["ols_edu2","ols_edu2"] + V["dml_nn","dml_nn"] - 2*V["ols_edu2","dml_nn"])

# ---- t-stats using point_est from (A) ----
t_ols_edu2_rf     <- (point_est["ols_edu2"] - point_est["dml_rf"])    / se_diff_ols_edu2_rf
t_ols_edu2_xg     <- (point_est["ols_edu2"] - point_est["dml_xgb"])   / se_diff_ols_edu2_xg
t_ols_edu2_lasso  <- (point_est["ols_edu2"] - point_est["dml_lasso"]) / se_diff_ols_edu2_lasso
t_ols_edu2_gam    <- (point_est["ols_edu2"] - point_est["dml_gam"])   / se_diff_ols_edu2_gam
t_ols_edu2_nn     <- (point_est["ols_edu2"] - point_est["dml_nn"])    / se_diff_ols_edu2_nn


# ---- two-sided p-values (edu2) ----
p_ols_edu2_rf     <- 2 * pnorm(-abs(t_ols_edu2_rf))
p_ols_edu2_xg     <- 2 * pnorm(-abs(t_ols_edu2_xg))
p_ols_edu2_lasso  <- 2 * pnorm(-abs(t_ols_edu2_lasso))
p_ols_edu2_gam    <- 2 * pnorm(-abs(t_ols_edu2_gam))
p_ols_edu2_nn     <- 2 * pnorm(-abs(t_ols_edu2_nn))



## ================
## Ols with college vs DML
## ================
# ---- SE of differences: OLS_COLL - DML ----
se_diff_ols_coll_rf     <- sqrt(V["ols_coll","ols_coll"] + V["dml_rf","dml_rf"] - 2*V["ols_coll","dml_rf"])
se_diff_ols_coll_xg     <- sqrt(V["ols_coll","ols_coll"] + V["dml_xgb","dml_xgb"] - 2*V["ols_coll","dml_xgb"])
se_diff_ols_coll_lasso  <- sqrt(V["ols_coll","ols_coll"] + V["dml_lasso","dml_lasso"] - 2*V["ols_coll","dml_lasso"])
se_diff_ols_coll_gam    <- sqrt(V["ols_coll","ols_coll"] + V["dml_gam","dml_gam"] - 2*V["ols_coll","dml_gam"])
se_diff_ols_coll_nn     <- sqrt(V["ols_coll","ols_coll"] + V["dml_nn","dml_nn"] - 2*V["ols_coll","dml_nn"])

# ---- t-stats using point_est from (A) ----
t_ols_coll_rf     <- (point_est["ols_coll"] - point_est["dml_rf"])    / se_diff_ols_coll_rf
t_ols_coll_xg     <- (point_est["ols_coll"] - point_est["dml_xgb"])   / se_diff_ols_coll_xg
t_ols_coll_lasso  <- (point_est["ols_coll"] - point_est["dml_lasso"]) / se_diff_ols_coll_lasso
t_ols_coll_gam    <- (point_est["ols_coll"] - point_est["dml_gam"])   / se_diff_ols_coll_gam
t_ols_coll_nn     <- (point_est["ols_coll"] - point_est["dml_nn"])    / se_diff_ols_coll_nn

# ---- two-sided p-values ----
p_ols_coll_rf     <- 2 * pnorm(-abs(t_ols_coll_rf))
p_ols_coll_xg     <- 2 * pnorm(-abs(t_ols_coll_xg))
p_ols_coll_lasso  <- 2 * pnorm(-abs(t_ols_coll_lasso))
p_ols_coll_gam    <- 2 * pnorm(-abs(t_ols_coll_gam))
p_ols_coll_nn     <- 2 * pnorm(-abs(t_ols_coll_nn))



summary_tbl <- data.frame(
  comparison = c(
    "OLS vs DML_RF", "OLS vs DML_XGB", "OLS vs DML_LASSO", "OLS vs DML_GAM", "OLS vs DML_NN",
    "OLS_edu2 vs DML_RF", "OLS_edu2 vs DML_XGB", "OLS_edu2 vs DML_LASSO", "OLS_edu2 vs DML_GAM", "OLS_edu2 vs DML_NN",
    "OLS_college vs DML_RF", "OLS_college vs DML_XGB", "OLS_college vs DML_LASSO", "OLS_college vs DML_GAM", "OLS_college vs DML_NN"
  ),
  SE = c(
    se_diff_ols_rf, se_diff_ols_xg, se_diff_ols_lasso, se_diff_ols_gam, se_diff_ols_nn,
    se_diff_ols_edu2_rf, se_diff_ols_edu2_xg, se_diff_ols_edu2_lasso, se_diff_ols_edu2_gam, se_diff_ols_edu2_nn,
    se_diff_ols_coll_rf, se_diff_ols_coll_xg, se_diff_ols_coll_lasso, se_diff_ols_coll_gam, se_diff_ols_coll_nn
  ),
  t_stat = c(
    t_ols_rf, t_ols_xg, t_ols_lasso, t_ols_gam, t_ols_nn,
    t_ols_edu2_rf, t_ols_edu2_xg, t_ols_edu2_lasso, t_ols_edu2_gam, t_ols_edu2_nn,
    t_ols_coll_rf, t_ols_coll_xg, t_ols_coll_lasso, t_ols_coll_gam, t_ols_coll_nn
  ),
  p_value = c(
    p_ols_rf, p_ols_xg, p_ols_lasso, p_ols_gam, p_ols_nn,
    p_ols_edu2_rf, p_ols_edu2_xg, p_ols_edu2_lasso, p_ols_edu2_gam, p_ols_edu2_nn,
    p_ols_coll_rf, p_ols_coll_xg, p_ols_coll_lasso, p_ols_coll_gam, p_ols_coll_nn
  )
)


# round
numeric_cols <- c("SE", "t_stat", "p_value")

summary_tbl[ , numeric_cols] <- round(summary_tbl[ , numeric_cols], 4)

print(summary_tbl)
