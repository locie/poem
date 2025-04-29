data {
  int<lower=1> T_train;      // num observations for training
  int<lower=1> T;            // num observations (total)
  int<lower=1> K;            // number of predictors
  
  array[T_train] real y;     // observed outputs for training
  matrix[T, K] x;            // inputs
}
parameters {
  real alpha;                // mean coeff
  vector[K] beta;        // exogenous
  real phi;                  // autoregression coeff
  real theta;                // moving avg coeff
  real<lower=0> sigma;       // noise scale
}
transformed parameters {
  vector[T_train] nu;              // prediction for time t
  vector[T_train] err;             // error for time t
  nu[1] = alpha + x[1]*beta + phi * (alpha + x[1]*beta);     // assume err[0] = 0
  err[1] = y[1] - nu[1];
  for (t in 2:T_train) {
    nu[t] = alpha + x[t]*beta + phi*y[t - 1] + theta*err[t - 1];
    err[t] = y[t] - nu[t];
  }
}
model {
  //mu ~ normal(0, 10);        // priors
  //phi ~ normal(0, 2);
  //theta ~ normal(0, 2);
  //sigma ~ cauchy(0, 5);
  err ~ normal(0, sigma);    // likelihood
  // y ~ normal(nu, sigma);    // c'est la même chose
}
generated quantities {

  vector[T] y_pred;
  // Prediction within the training dataset
  for (t in 1:T_train) {
    y_pred[t] = normal_rng(nu[t], sigma);
  }
  // Forecasts with increased uncertainty due to the autocorrelation
  for (t in T_train+1:T) {
    y_pred[t] = normal_rng(alpha + x[t]*beta + phi*y_pred[t-1], sigma*sqrt(1+theta^2));
  }
}
