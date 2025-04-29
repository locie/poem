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
  real rho1;     // autoregressive coefficient
  real rho2;
  real<lower=0> sigma;  // error scale
}
model {
  // Priors
  a1 ~ normal(a1_prior[1], a1_prior[2]);
  a2 ~ normal(a2_prior[1], a2_prior[2]);
  x1 ~ normal(x1_prior[1], x1_prior[2]);
  y1 ~ normal(y1_prior[1], y1_prior[2]);
  rho1 ~ normal(0, 0.1);
  rho2 ~ normal(0, 0.1);
  // Observational model
  // voir http://herbsusmann.com/2019/08/09/autoregressive-processes-are-gaussian-processes/ pour la première ligne
  // ATTENTION je crois que le sigma^2/(1-rho^2) est faux
  // y[1] ~ normal(power_mean(x[1], a1, a2, x1, y1), sqrt(sigma^2 / (1-rho^2)));
  // y[1] ~ normal(power_mean(x[1], a1, a2, x1, y1) * (1+rho), sigma);
  for (n in 3:N_train) {
    y[n] ~ normal(power_mean(x[n], a1, a2, x1, y1) + rho1*y[n-1] + rho2*y[n-2], sigma);
  }
}
generated quantities {
  vector[N_train] log_lik;
  vector[N] prediction;
  vector[N-N_train] psi;
  vector[N-N_train] newsigma;
  
  // One-step ahead prediction during the training period
  prediction[1] = normal_rng(power_mean(x[1], a1, a2, x1, y1) * (1+rho1+rho2), sigma);
  log_lik[1] = normal_lpdf(y[1] | power_mean(x[1], a1, a2, x1, y1) * (1+rho1+rho2), sigma);
  prediction[2] = normal_rng(power_mean(x[2], a1, a2, x1, y1) + (rho1+rho2)*y[1], sigma);
  log_lik[2] = normal_lpdf(y[2] | power_mean(x[2], a1, a2, x1, y1) + (rho1+rho2)*y[1], sigma);
  for (n in 3:N_train) {
    prediction[n] = normal_rng(power_mean(x[n], a1, a2, x1, y1) + rho1*y[n-1] + rho2*y[n-2], sigma);
    log_lik[n] = normal_lpdf(y[n] | power_mean(x[n], a1, a2, x1, y1) + rho1*y[n-1] + rho2*y[n-2], sigma);
  }
  // Forecasting after the training period
  psi[1] = 1;
  newsigma[1] = sigma;
  prediction[N_train+1] = normal_rng(power_mean(x[N_train+1], a1, a2, x1, y1) + rho1*y[N_train] + rho2*y[N_train-1], newsigma[1]);
  // Attention les indices de psi sont décalés : psi[1] = psi_0 ; psi[2] = psi_1 etc.
  psi[2] = rho1 * psi[1];
  newsigma[2] = sigma * sqrt(1+psi[2]^2);
  prediction[N_train+2] = normal_rng(power_mean(x[N_train+2], a1, a2, x1, y1) + rho1*prediction[N_train+1] + rho2*y[N_train], newsigma[2]);
  for (n in N_train+3:N) {
    psi[n-N_train] = rho1 * psi[n-N_train-1] + rho2 * psi[n-N_train-2];
    newsigma[n-N_train] = sigma * sqrt(sum(psi[1:(n-N_train)]^2));
    prediction[n] = normal_rng(power_mean(x[n], a1, a2, x1, y1) + rho1*prediction[n-1] + rho2*prediction[n-2], newsigma[n-N_train]);
  }
}