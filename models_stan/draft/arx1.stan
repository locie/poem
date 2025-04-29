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
  real<lower=0> sigma;       // noise scale
}

model {
  //mu ~ normal(0, 10);        // priors
  //phi ~ normal(0, 2);
  //theta ~ normal(0, 2);
  //sigma ~ cauchy(0, 5);
  for (t in 2:T_train) {
    y[t] ~ normal(alpha + x[t]*beta + phi*y[t-1], sigma);
  }
}
generated quantities {

  vector[T] y_pred;
  // Prediction within the training dataset
  // y_pred[1] = (alpha + x[1]*beta) * (1+phi);
  y_pred[1] = y[1];
  for (t in 2:T_train) {
    y_pred[t] = normal_rng(alpha + x[t]*beta + phi*y[t-1], sigma);
  }
  // Forecasts with increased uncertainty due to the autocorrelation
  for (t in T_train+1:T) {
    y_pred[t] = normal_rng(alpha + x[t]*beta + phi*y_pred[t-1], sigma);
  }
}
