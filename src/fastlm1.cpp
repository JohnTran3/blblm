#include <RcppArmadillo.h>
//[[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;
//'@title this is the title
//' Fastlm function in rcpp
//'@param X matrix
//'@param y vector
//'@param w vector
//'@export fastLm


//[[Rcpp::export]]
List fastLm(const arma::mat& X, const arma::colvec& y, const arma::vec& w){
  int n = X.n_rows, k = X.n_cols, r = rank(X);

  arma::mat weights = diagmat(w);
  arma::colvec coef = arma::inv(arma::trans(X)*weights*X)*arma::trans(X)*weights*y;
  arma::colvec res  = y - X*coef;
  // std.errors of coefficients
  double s2 = std::inner_product(res.begin(), res.end(), res.begin(), 0.0)/(n - k);




  return Rcpp::List::create(Rcpp::Named("coefficients") = coef,
                            Rcpp::Named("rank") = r,
                            Rcpp::Named("weights") = w,
                            Rcpp::Named("residuals")= res,
                            Rcpp::Named("response") = y,
                            Rcpp::Named("df.residual")  = n - k);



}



