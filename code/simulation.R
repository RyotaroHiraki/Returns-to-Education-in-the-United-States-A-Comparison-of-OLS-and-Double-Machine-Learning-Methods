## ======== packages ========
suppressPackageStartupMessages({
  library(DoubleML)
  library(mlr3)
  library(mlr3learners)   # regr.ranger / regr.xgboost / regr.cv_glmnet
  library(mlr3pipelines)
  library(data.table)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(mgcv)
  library(nnet)
  library(tibble)
})

set.seed(123)

## ======== simulation settings ========
n_sim <- 100
n     <- 1000
true_ate <- 0.1
K <- 5

## ======== DGPs ========
# Case 1: g(X) is linear  -> OLS correct
gen_data_linear <- function(n, beta = 0.1){
  X1 <- rnorm(n); X2 <- rnorm(n); X3 <- rnorm(n); X4 <- rnorm(n)
  # W linear in X (exogenous)
  W  <- 12 + 0.8*X1 - 0.4*X2 + 0.6*X3 + 0.3*X4 + rnorm(n, sd=1)
  education <- pmin(20, pmax(8, round(W)))
  # g(X) linear
  eps <- rnorm(n, sd=1)
  wage <- 1.0 + beta*education + 0.6*X1 - 0.5*X2 + 0.4*X3 + 0.3*X4 + eps
  data.frame(wage, education, X1, X2, X3, X4)
}

# Case 2: g(X) is nonlinear -> OLS misspecified; DML should handle
gen_data_nonlinear <- function(n, beta = 0.1){
  X1 <- rnorm(n); X2 <- rnorm(n); X3 <- rnorm(n); X4 <- rnorm(n)
  S3 <- as.integer(X3 > 0)    # step(X3)
  # W can be linear (keeps focus on g(X)); still exogenous
  W  <- 12 + 0.8*X1 - 0.4*X2 + 0.6*X3 + 0.3*X4 + rnorm(n, sd=1)
  education <- pmin(20, pmax(8, round(W)))
  # g(X) nonlinear in original X's
  eps <- rnorm(n, sd=1)
  wage <- 1.0 + beta*education +
          0.6*X1 + 0.5*X2^2 - 0.4*X1*X2 + 0.25*S3 + 0.2*X4^3 + eps
  data.frame(wage, education, X1, X2, X3, X4, S3)
}

# Case 3: W and g(X) are nonlinear -> OLS misspecified; DML should handle
gen_data_nonlinear2 <- function(n, beta = 0.1){
  X1 <- rnorm(n); X2 <- rnorm(n); X3 <- rnorm(n); X4 <- rnorm(n)
  S3 <- as.integer(X3 > 0)    # step(X3)
  # W can be linear (keeps focus on g(X)); still exogenous
  W  <- 12 + 0.6*X1 + 0.5*X2^2 - 0.4*X1*X2 + 0.25*S3 + 0.2*X4^3 + rnorm(n, sd=1)
  education <- pmin(20, pmax(8, round(W)))
  # g(X) nonlinear in original X's
  eps <- rnorm(n, sd=1)
  wage <- 1.0 + beta*education +
          0.6*X1 + 0.5*X2^2 - 0.4*X1*X2 + 0.25*S3 + 0.2*X4^3 + eps
  data.frame(wage, education, X1, X2, X3, X4, S3)
}

## ======== helper: run one replication for a given generator ========
# Returns a named numeric vector:
# c(OLS, DML_RF, DML_XGB, DML_Lasso, DML_GAM, DML_NN)
run_one <- function(dat, K = 5, trees = 200){
  # --- OLS: only linear controls ---
  ols <- lm(wage ~ education + X1 + X2 + X3 + X4, data = dat)
  ols_hat <- unname(coef(ols)["education"])

  # --- DML-PLR: learners = RF / XGBoost / Lasso（give raw Xs） ---
  x_cols <- intersect(colnames(dat), c("X1","X2","X3","X4","S3"))
  dml_data <- DoubleMLData$new(
    data   = data.table::data.table(dat),
    y_col  = "wage",
    d_cols = "education",
    x_cols = x_cols
  )

  # RF
  rf <- mlr3::lrn("regr.ranger", num.trees = trees, mtry.ratio = 0.6, min.node.size = 5, num.threads = 1, respect.unordered.factors = "order")
  mdl_rf <- DoubleMLPLR$new(dml_data, ml_g = rf, ml_m = rf, n_folds = K)
  mdl_rf$fit()
  rf_hat <- as.numeric(mdl_rf$coef)

  # XGBoost
  xg <- mlr3::lrn("regr.xgboost", nrounds = 200, eta = 0.3, max_depth = 6, subsample = 0.8, objective = "reg:squarederror",
                   nthread = 1)
  mdl_xg <- DoubleMLPLR$new(dml_data, ml_g = xg, ml_m = xg, n_folds = K)
  mdl_xg$fit()
  xg_hat <- as.numeric(mdl_xg$coef)

  # Lasso（cv_glmnet）
  # define pipeline: scale → Lasso learner
  scale_pipe <- po("scale")
  lasso_learner <- lrn("regr.cv_glmnet",
                     alpha = 1,
                     nfolds = 10,
                     standardize = FALSE)   # disable internal scaling

  lasso_pipe <- GraphLearner$new(scale_pipe %>>% lasso_learner)

  mdl_ls <- DoubleMLPLR$new(dml_data,
                          ml_g = lasso_pipe,
                          ml_m = lasso_pipe,
                          n_folds = K)

  mdl_ls$fit()
  lasso_hat <- as.numeric(mdl_ls$coef)

  # --- GAM（manually made partialling-out + cross validation）---
  n <- nrow(dat)
  folds <- sample(rep(1:K, length.out = n))
  gam_res_y <- numeric(n)
  gam_res_d <- numeric(n)

  s_terms <- c("s(X1)", "s(X2)", "s(X3)", "s(X4)")
  if ("S3" %in% names(dat)) s_terms <- c(s_terms, "S3")
  f_y <- as.formula(paste("wage ~", paste(s_terms, collapse = " + ")))
  f_d <- as.formula(paste("education ~", paste(s_terms, collapse = " + ")))

  for (k in 1:K) {
    train_idx <- which(folds != k)
    test_idx  <- which(folds == k)

    train <- dat[train_idx, ]
    test  <- dat[test_idx, ]

    # g_hat(X) = E[Y|X]
    gam_y <- mgcv::gam(f_y, data = train, method = 'REML')
    ghat  <- predict(gam_y, newdata = test)
    gam_res_y[test_idx] <- test$wage - ghat

    # m_hat(X) = E[W|X]
    gam_d <- mgcv::gam(f_d, data = train, method = 'REML')
    mhat  <- predict(gam_d, newdata = test)
    gam_res_d[test_idx] <- test$education - mhat

  }
  gam_hat <- coef(lm(gam_res_y ~ gam_res_d - 1))[1]

  # --- Neural Net（manually made partialling-out + cross-validation）---
  folds2 <- sample(rep(1:K, length.out = n))
  nn_res_y <- numeric(n)
  nn_res_d <- numeric(n)

  for (k in 1:K) {
    train_idx <- which(folds2 != k)
    test_idx  <- which(folds2 == k)

    # （handle the case where S3 doesnt exist）
    nn_cols <- intersect(c("wage","education","X1","X2","X3","X4","S3"), names(dat))
    train <- dat[train_idx, nn_cols, drop = FALSE]
    test  <- dat[test_idx,  nn_cols, drop = FALSE]

    # scaling（Exclude S3 as it is dummy）
    scale_cols <- intersect(c("wage","education","X1","X2","X3","X4"), colnames(train))
    m <- sapply(train[, scale_cols, drop = FALSE], mean)
    s <- pmax(sapply(train[, scale_cols, drop = FALSE], sd), 1e-8)
    for (nm in scale_cols) {
      train[[nm]] <- (train[[nm]] - m[[nm]])/s[[nm]]
      test[[nm]]  <- (test[[nm]]  - m[[nm]])/s[[nm]]
    }

    # y ~ controls（educationは入れない）
    f_nn_y <- as.formula(paste("wage ~", paste(setdiff(nn_cols, c("wage","education")), collapse = " + ")))
    nn_y <- nnet::nnet(f_nn_y, data = train, linout = TRUE, size = 5, maxit = 500, trace = FALSE, decay = 0.01, maxNWts = 5000)
    ghat <- predict(nn_y, newdata = test)
    nn_res_y[test_idx] <- as.numeric(test$wage - ghat)

    # d ~ controls
    f_nn_d <- as.formula(paste("education ~", paste(setdiff(nn_cols, c("wage","education")), collapse = " + ")))
    nn_d <- nnet::nnet(f_nn_d, data = train, linout = TRUE, size = 5, maxit = 500, trace = FALSE, decay = 0.01, maxNWts = 5000)
    mhat <- predict(nn_d, newdata = test)
    nn_res_d[test_idx] <- as.numeric(test$education - mhat)

  }
  nn_hat <- coef(lm(nn_res_y ~ nn_res_d - 1))[1]

  # return all together
  c(
    OLS        = ols_hat,
    DML_RF     = rf_hat,
    DML_XGB    = xg_hat,
    DML_Lasso  = lasso_hat,
    DML_GAM    = gam_hat,
    DML_NN     = nn_hat
  )
}


## ======== Monte Carlo ========
sim_case <- function(gen_fun, label){
  res <- map_dfr(1:n_sim, function(i){
      if(i %% 5 == 0){
    message(paste0("Iteration #:", i, "for", label))
    }
    dat <- gen_fun(n, beta = true_ate)
    est <- run_one(dat)
    tibble(sim = i,
           estimator = names(est),
           theta_hat = as.numeric(est),
           case = label)
  })
  res
}

res_lin  <- sim_case(gen_data_linear,   "Case 1: linear g(X)")
res_nlin <- sim_case(gen_data_nonlinear,"Case 2: nonlinear g(X)")
res_2nlin <- sim_case(gen_data_nonlinear2, "Case 3: non linear W and g(x)")
res_all <- bind_rows(res_lin, res_nlin, res_2nlin) %>%
  mutate(true = true_ate,
         bias = theta_hat - true)

## ======== summaries ========
summary_table <- res_all %>%
  group_by(case, estimator) %>%
  summarise(mean_hat = mean(theta_hat),
            sd_hat   = sd(theta_hat),
            bias     = mean(theta_hat - true),
            rmse     = sqrt(mean((theta_hat - true)^2)),
            .groups = "drop") %>%
  arrange(case, rmse)

print(summary_table)


##===== Box plot========
library(ggplot2)
library(dplyr)

#Case 1
res_case1 <- res_all %>%
  filter(case == "Case 1: linear g(X)")

ggplot(res_case1, aes(x = estimator, y = theta_hat, fill = estimator)) +
  geom_boxplot(alpha = 0.7, outlier.shape = 21, outlier.size = 1.5) +
  geom_hline(yintercept = true_ate, linetype = "dashed", color = "red", linewidth = 0.8) +
  theme_bw(base_size = 12) +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 30, hjust = 1)
  ) +
    scale_x_discrete(
    labels= c(
      'OLS' = 'OLS',
      'DML_RF' = 'DML: RF',
      'DML_XGB' = 'DML: XGboost',
      'DML_Lasso' = 'DML: Lasso',
      'DML_GAM.gam_res_d'= 'DML: GAMs',
      'DML_NN.nn_res_d' = 'DML: Neural Nets'
    )
  ) +
  labs(
    title = "Simulation 1: Linear g(X)",
    x = "Estimator",
    y = "Estimated coefficient (ATE)",
    caption = "Red dashed line = True ATE (0.1)"
  )

#Case2
res_case2 <- res_all %>%
  filter(case == "Case 2: nonlinear g(X)")

ggplot(res_case2, aes(x = estimator, y = theta_hat, fill = estimator)) +
  geom_boxplot(alpha = 0.7, outlier.shape = 21, outlier.size = 1.5) +
  geom_hline(yintercept = true_ate, linetype = "dashed", color = "red", linewidth = 0.8) +
  theme_bw(base_size = 12) +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 30, hjust = 1)
  ) +
  scale_x_discrete(
    labels= c(
      'OLS' = 'OLS',
      'DML_RF' = 'DML: RF',
      'DML_XGB' = 'DML: XGboost',
      'DML_Lasso' = 'DML: Lasso',
      'DML_GAM.gam_res_d'= 'DML: GAMs',
      'DML_NN.nn_res_d' = 'DML: Neural Nets'
    )
  ) +
  labs(
    title = "Simulation 2: Nonlinear g(X)",
    x = "Estimator",
    y = "Estimated coefficient (ATE)",
    caption = "Red dashed line = True ATE (0.1)"
  )

#Case3
res_case3 <- res_all %>%
  filter(case == "Case 3: non linear W and g(x)")

ggplot(res_case3, aes(x = estimator, y = theta_hat, fill = estimator)) +
  geom_boxplot(alpha = 0.7, outlier.shape = 21, outlier.size = 1.5) +
  geom_hline(yintercept = true_ate, linetype = "dashed", color = "red", linewidth = 0.8) +
  theme_bw(base_size = 12) +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 30, hjust = 1)
  ) +
    scale_x_discrete(
    labels= c(
      'OLS' = 'OLS',
      'DML_RF' = 'DML: RF',
      'DML_XGB' = 'DML: XGboost',
      'DML_Lasso' = 'DML: Lasso',
      'DML_GAM.gam_res_d'= 'DML: GAMs',
      'DML_NN.nn_res_d' = 'DML: Neural Nets'
    )
  ) +
  labs(
    title = "Simulation 3: Nonlinear W and g(X)",
    x = "Estimator",
    y = "Estimated coefficient (ATE)",
    caption = "Red dashed line = True ATE (0.1)"
  )