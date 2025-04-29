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
  vector[2] theta_prior;
}
parameters {
  real a1;      // first slope
  real a2;      // second slope
  real x1;      // change point temperature
  real y1;      // change point energy
  real<lower=0> sigma;  // error scale
  real theta;   // moving average
}
transformed parameters {
  vector[N_train] f;
  vector[N_train] epsilon;
  for (n in 1:N_train) {
    f[n] = power_mean(x[n], a1, a2, x1, y1);
  }
  epsilon[1] = y[1] - f[1];
  for (n in 2:N_train) {
    epsilon[n] = y[n] - (f[n] + theta*epsilon[n-1]);
  }

}
model {
  // Priors
  a1 ~ normal(a1_prior[1], a1_prior[2]);
  a2 ~ normal(a2_prior[1], a2_prior[2]);
  x1 ~ normal(x1_prior[1], x1_prior[2]);
  y1 ~ normal(y1_prior[1], y1_prior[2]);
  theta ~ normal(theta_prior[1], theta_prior[2]);
  // Observational model
  for (n in 2:N_train) {
    // y[n] ~ normal(f[n] + theta*epsilon[n-1], sigma);
    epsilon[n] ~ normal(0, sigma);
  }
}
generated quantities {
  vector[N] prediction;
  vector[N_train] log_lik;
  
  // One-step ahead prediction during the training period
  prediction[1] = normal_rng(f[1], sigma);
  log_lik[1] = normal_lpdf(y[1] | f[1], sigma);
  for (n in 2:N_train) {
    prediction[n] = normal_rng(f[n] + theta*epsilon[n-1], sigma);
    log_lik[n] = normal_lpdf(y[n] | f[n] + theta*epsilon[n-1], sigma);
  }
  
  // Forecasting after the training period with increased uncertainty due to the autocorrelation
  prediction[N_train+1] = normal_rng(power_mean(x[N_train+1], a1, a2, x1, y1) + theta*epsilon[N_train], sigma);
  for (m in N_train+2:N) {
    prediction[m] = normal_rng(power_mean(x[m], a1, a2, x1, y1), sigma*sqrt(1+theta^2));
  }
}