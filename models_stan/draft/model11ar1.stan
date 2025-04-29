functions {
  real power_mean(real x, real a1, real a2, real x1, real y1) {
    return x <= x1 ? y1 - a1*(x1-x) : y1 + a2*(x-x1) ;
  }
}
data {
  int<lower=0> N_train;  // number of training data points
  int<lower=0> N;       // number of data items
  vector[N] x;          // outdoor temperature
  vector[N_train] y;      // outcome energy vector
  
  vector[2] a1_prior;
  vector[2] a2_prior;
  vector[2] x1_prior;
  vector[2] y1_prior;
  // vector[2] rho_prior;
}
parameters {
  real a1;      // first slope
  real a2;      // second slope
  real x1;      // change point temperature
  real y1;      // change point energy
  real rho;     // autoregressive coefficient
  real<lower=0> sigma;  // error scale
}
model {
  // Priors
  a1 ~ normal(a1_prior[1], a1_prior[2]);
  a2 ~ normal(a2_prior[1], a2_prior[2]);
  x1 ~ normal(x1_prior[1], x1_prior[2]);
  y1 ~ normal(y1_prior[1], y1_prior[2]);
  rho ~ normal(0, 0.1);
  // Observational model
  // voir http://herbsusmann.com/2019/08/09/autoregressive-processes-are-gaussian-processes/ pour la première ligne
  // ATTENTION je crois que le sigma^2/(1-rho^2) est faux
  // y[1] ~ normal(power_mean(x[1], a1, a2, x1, y1), sqrt(sigma^2 / (1-rho^2)));
  // y[1] ~ normal(power_mean(x[1], a1, a2, x1, y1) * (1+rho), sigma);
  for (n in 2:N_train) {
    y[n] ~ normal(power_mean(x[n], a1, a2, x1, y1) + rho*y[n-1], sigma);
  }
}
generated quantities {
  vector[N_train] log_lik;
  vector[N] prediction;
  // One-step ahead prediction during the training period
  prediction[1] = normal_rng(power_mean(x[1], a1, a2, x1, y1) * (1+rho), sigma);
  log_lik[1] = normal_lpdf(y[1] | power_mean(x[1], a1, a2, x1, y1) * (1+rho), sigma);
  for (n in 2:N_train) {
    prediction[n] = normal_rng(power_mean(x[n], a1, a2, x1, y1) + rho*y[n-1], sigma);
    log_lik[n] = normal_lpdf(y[n] | power_mean(x[n], a1, a2, x1, y1) + rho*y[n-1], sigma);
  }
  // Forecasting after the training period
  prediction[N_train+1] = normal_rng(power_mean(x[N_train+1], a1, a2, x1, y1) + rho*y[N_train], sigma);
  for (m in N_train+2:N) {
    prediction[m] = normal_rng(power_mean(x[m], a1, a2, x1, y1) + rho*prediction[m-1], sigma * sqrt( 1+rho^2*(1-rho^(2*(m-N_train-1))) / (1-rho^2) ));
  }
}