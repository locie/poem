functions {
  // This chunk is the formula for the changepoint model which will be used several times in this program
  real power_mean(real T, real alpha, real beta_h, real beta_c, real tau_h, real tau_c) {
    return (alpha + beta_h * fmax(tau_h-T, 0) + beta_c * fmax(T-tau_c, 0));
  }
}

data {
  // This block declares all data which will be passed to the Stan model.
  int<lower=0> N;            // number of data items in the baseline period
  int K;              // number of possible states
  real T[N];           // temperature (baseline)
  real y[N];           // outcome energy vector (baseline)
  int weekend[N];       // categorical variable for the week ends (baseline)
  int hour[N];       // hour (baseline)
  real prior_alpha[K];
  real prior_beta_h[K];
  real prior_beta_c[K];
  real prior_tau_h[K];
  real prior_tau_c[K];
}
parameters {
  // This block declares the parameters of the model.
  simplex[K] theta[24, 2, K];     // transition probability matrix
  positive_ordered[K] alpha;         // baseline consumption
  positive_ordered[K] beta_h;        // slopes for heating
  positive_ordered[K] beta_c;        // slopes for cooling
  real tau_h[K];
  real tau_c[K];   // threshold temperatures (week end and working days)
  real<lower=0> sigma;         // error scale
}

model {
  // ici on estime les paramètres K = 2 états possibles
  real acc[K];          // components of the forward variable being calculated
  real gamma[N, K];     // table of forward variables: log marginal prob of the inputs up to time t given latent state k
  
  // priors: paramètres de l'émission. inoccupé et occupé
  alpha ~ normal(prior_alpha, [1, 1]);
  beta_h ~ normal(prior_beta_h, [0.2, 0.2]);
  beta_c ~ normal(prior_beta_c, [0.3, 0.3]);
  tau_h ~ normal(prior_tau_h, [1.5, 1.5]);
  tau_c ~ normal(prior_tau_c, [3, 1]);
  sigma ~ normal(5, 1);
  // priors: matrice de transition
  for (j in 1:K) {
    theta[:5, 1, j, 1] ~ beta(rep_array(5.0, 5), rep_array(1.0, 5));    // inoccupé le matin en semaine
    theta[6:18, 1, j, 2] ~ beta(rep_array(5.0, 13), rep_array(1.0, 13));  // occupé en journée en semaine
    theta[19:, 1, j, 1] ~ beta(rep_array(5.0, 6), rep_array(1.0, 6));   // inoccupé le soir en semaine
    theta[:, 2, j, 1] ~ beta(rep_array(5.0, 24), rep_array(1.0, 24));     // inoccupé tout le week end
  }

  // forward algorithm
  {
  for (k in 1:K) {
    gamma[1, k] = normal_lpdf(y[1] | power_mean(T[1], alpha[k], beta_h[k], beta_c[k], tau_h[k], tau_c[k]), sigma);
  }
  for (t in 2:N) {
    int h = hour[t] + 1;
    int w = weekend[t] + 1;
    for (k in 1:K) {
      for (j in 1:K)
        acc[j] = gamma[t-1, j] + log(theta[h, w, j, k]) + normal_lpdf(y[t] | power_mean(T[t], alpha[k], beta_h[k], beta_c[k], tau_h[k], tau_c[k]), sigma);
      gamma[t, k] = log_sum_exp(acc);
    }
  }
  target += log_sum_exp(gamma[N]);
  }
}

generated quantities {

  // Viterbi pour estimer les états d'occupation sur la période reporting
  // puis prédiction sur reporting avec les paramètres baseline
  
  int<lower=1, upper=K> y_star[N];
  real log_p_y_star;
  {
    int back_ptr[N, K];
    real best_logp[N, K];
    real best_total_logp;
    for (k in 1:K) {
      best_logp[1, k] = normal_lpdf(y[1] | power_mean(T[1], alpha[k], beta_h[k], beta_c[k], tau_h[k], tau_c[k]), sigma);
    }
    for (t in 2:N) {
      int h = hour[t] + 1;
      int w = weekend[t] + 1;
      for (k in 1:K) {
        best_logp[t, k] = negative_infinity();
        for (j in 1:K) {
          real logp;
          logp = best_logp[t - 1, j] + log(theta[h, w, j, k]) + normal_lpdf(y[t] | power_mean(T[t], alpha[k], beta_h[k], beta_c[k], tau_h[k], tau_c[k]), sigma);
          if (logp > best_logp[t, k]) {
            back_ptr[t, k] = j;
            best_logp[t, k] = logp;
          }
        }
      }
    }
    log_p_y_star = max(best_logp[N]);
    for (k in 1:K) {
      if (best_logp[N, k] == log_p_y_star) {
        y_star[N] = k;
      }
    }
    for (t in 1:(N - 1)) {
      y_star[N - t] = back_ptr[N - t + 1, y_star[N - t + 1]];
    }
  }
}