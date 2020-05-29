data {
  int<lower=0> N;    // number of samples
  int<lower=0> R;    // number of runs (batches)
  int<lower=0> D;    // number of dimensions
  int<lower=0> J;    // number of subjects
  int<lower=0> p;    // number of covariates
  real depth[N];     // sequencing depths of microbes
  matrix[N, p] x;    // covariate matrix
  int y[N, D];       // observed microbe abundances
  int<lower=1, upper=J> subj_ids[N];   // subject ids
  int<lower=1, upper=R> batch_ids[N];  // batch ids
  matrix[R, D-1] gamma;                // batch differnetials
}

parameters {
  // parameters required for linear regression on the species means
  matrix[p, D-1] beta;                 // covariates
  matrix[J, D-1] alpha;                // subject differentials
  real<lower=0.01> disp;
}

transformed parameters {
  matrix[N, D-1] lam;
  matrix[N, D] lam_clr;
  matrix[N, D] prob;
  vector[N] z;

  z = to_vector(rep_array(0, N));
  lam = x * beta;
  // add batch effects
  for (n in 1:N){
    lam[n] += gamma[batch_ids[n]];
  }
  // add in subject specific effects
  for (n in 1:N){
    lam[n] += alpha[subj_ids[n]];
  }
  lam_clr = append_col(z, lam);
}

model {
  // setting priors ...
  disp ~ inv_gamma(1., 1.);
  for (i in 1:D-1){
    for (j in 1:p){
      beta[j, i] ~ normal(0., 5.); // uninformed prior
    }
  }
  // generating counts
  for (n in 1:N){
    for (i in 1:D){
      target += neg_binomial_2_log_lpmf(y[n, i] | depth[n] + lam_clr[n, i], disp);
    }
  }
}