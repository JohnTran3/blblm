#' @import purrr
#' @import stats
#' @import furrr
#' @import RcppArmadillo
#' @import tidyverse
#' @import utils
#' @importFrom magrittr %>%
#' @details
#' Linear Regression with Little Bag of Bootstraps
"_PACKAGE"


## quiets concerns of R CMD check re: the .'s that appear in pipelines
# from https://github.com/jennybc/googlesheets/blob/master/R/googlesheets.R
utils::globalVariables(c("."))


#' @export
#' @param formula formula for the linear regression
#' @param data dataset to input
#' @param m amount of splits
#' @param B bootstraps
#' @param files files list
#' @param no_cores number of cores
blblm <- function(formula, data, m = 10, B = 5000,no_cores = 1,files = NULL) {
  if(is.null(files)){
    data_list <- split_data(data, m)
  }else{
    data_list <- list()
    for(i in 1:length(files)){
      name <- paste(i)
      tmp <- read_csv(files[i])
      data_list[[name]] <- tmp
  }}
  if(no_cores == 1){
    estimates <- map(
      data_list,
      ~ lm_each_subsample(formula = formula, data = ., n = nrow(data), B = B))
    }else{
    plan(multicore,workers= no_cores)
  #data_list <- split_data(data, m)
    estimates <- future_map(
    data_list,
    ~ lm_each_subsample(formula = formula, data = ., n = nrow(data), B = B))}

  res <- list(estimates = estimates, formula = formula)
  class(res) <- "blblm"
  invisible(res)
}

#' @title split_data
#' @param data dataset
#' @param m index
#' split data into m parts of approximated equal sizes
split_data <- function(data, m) {
  idx <- sample.int(m, nrow(data), replace = TRUE)
  data %>% split(idx)
}

#'@title lm_each_subsample
#'@param formula linear regression model
#'@param data dataset
#'@param n number of n
#'@param B bootstrap
#' compute the estimates
lm_each_subsample <- function(formula, data, n, B) {
  replicate(B, lm_each_boot(formula, data, n), simplify = FALSE)
}

#' @title lm_each_boot
#' @param formula  linear model
#' @param data dataset
#' @param n length of freqs
#'
#' compute the regression estimates for a blb dataset
lm_each_boot <- function(formula, data, n) {
  freqs <- rmultinom(1, n, rep(1, nrow(data)))
  lm1(formula, data, freqs)
}

#' @title lm1
#' @param formula formula
#' @param data dataset
#' @param freqs weights
#' @return returns list of coef and sigma
#' estimate the regression estimates based on given the number of repetitions
lm1 <- function(formula, data, freqs) {
  # drop the original closure of formula,
  # otherwise the formula will pick a wront variable from the global scope.
  environment(formula) <- environment()
  x<-model.matrix(formula,data)
  y<-model.response(model.frame(formula,data))
  fit <- fastLm(x, y, w = freqs)
  list(coef = blbcoef(fit), sigma = blbsigma(fit))
}
#'@export
#'@title blbcoef
#'@param fit model
#'@return coef
#' compute the coefficients from fit
blbcoef <- function(fit) {
  coef(fit)
}

#' @export
#' @title blbsigma
#' @param fit model
#' @return
#' compute sigma from fit
blbsigma <- function(fit) {
  p <- fit$rank
  y <- fit$response
  e <- fit$residuals - y
  w <- fit$weights
  sqrt(sum(w * (e^2)) / (sum(w) - p))
}


#' @export
#' @method print blblm
print.blblm <- function(x, ...) {
  cat("blblm model:", capture.output(x$formula))
  cat("\n")
}


#' @export
#' @method sigma blblm
sigma.blblm <- function(object, confidence = FALSE, level = 0.95, ...) {
  est <- object$estimates
  sigma <- mean(map_dbl(est, ~ mean(map_dbl(., "sigma"))))
  if (confidence) {
    alpha <- 1 - 0.95
    limits <- est %>%
      map_mean(~ quantile(map_dbl(., "sigma"), c(alpha / 2, 1 - alpha / 2))) %>%
      set_names(NULL)
    return(c(sigma = sigma, lwr = limits[1], upr = limits[2]))
  } else {
    return(sigma)
  }
}

#' @export
#' @method coef blblm
coef.blblm <- function(object, ...) {
  est <- object$estimates
  map_mean(est, ~ map_cbind(., "coef") %>% rowMeans())
}


#' @export
#' @method confint blblm
confint.blblm <- function(object, parm = NULL, level = 0.95, ...) {
  if (is.null(parm)) {
    parm <- attr(terms(fit$formula), "term.labels")
  }
  alpha <- 1 - level
  est <- object$estimates
  out <- map_rbind(parm, function(p) {
    map_mean(est, ~ map_dbl(., list("coef", p)) %>% quantile(c(alpha / 2, 1 - alpha / 2)))
  })
  if (is.vector(out)) {
    out <- as.matrix(t(out))
  }
  dimnames(out)[[1]] <- parm
  out
}

#' @export
#' @method predict blblm
predict.blblm <- function(object, new_data, confidence = FALSE, level = 0.95, ...) {
  est <- object$estimates
  X <- model.matrix(reformulate(attr(terms(object$formula), "term.labels")), new_data)
  if (confidence) {
    map_mean(est, ~ map_cbind(., ~ X %*% .$coef) %>%
      apply(1, mean_lwr_upr, level = level) %>%
      t())
  } else {
    map_mean(est, ~ map_cbind(., ~ X %*% .$coef) %>% rowMeans())
  }
}


mean_lwr_upr <- function(x, level = 0.95) {
  alpha <- 1 - level
  c(fit = mean(x), quantile(x, c(alpha / 2, 1 - alpha / 2)) %>% set_names(c("lwr", "upr")))
}

map_mean <- function(.x, .f, ...) {
  (map(.x, .f, ...) %>% reduce(`+`)) / length(.x)
}

map_cbind <- function(.x, .f, ...) {
  map(.x, .f, ...) %>% reduce(cbind)
}

map_rbind <- function(.x, .f, ...) {
  map(.x, .f, ...) %>% reduce(rbind)
}
