data {
  int<lower=0> M;                // number of batch samples
  int<lower=0> R;                // number of runs (batches)
  int<lower=0> D;                // number of microbes
  int<lower=0> S;                // number of sample ids
  real depth[M];                 // sequencing depths of microbes

  int<lower=1, upper=S> samp_ids[M];   // sample ids
  int<lower=1, upper=R> batch_ids[M];  // batch ids
  int y[M, D];                         // observed microbe abundances
}

parameters {
  // parameters required for linear regression on the species means
  matrix[S, D-1] sdiff;                // sample differentials
  matrix[R, D-1] bdiff;                // batch differentials
  real<lower=0.01> disp;
}

transformed parameters {
  matrix[M, D-1] lam;
  matrix[M, D] lam_clr;
  matrix[M, D] prob;
  vector[M] z;

  z = to_vector(rep_array(0, M));

  // add batch effects and sample effects
  for (m in 1:M){
    lam[m] = sdiff[samp_ids[m]] + bdiff[batch_ids[m]];
  }

  lam_clr = append_col(z, lam);

}

model {

  // setting priors ...
  disp ~ inv_gamma(1., 1.);
  for (s in 1:S){
    for (d in 1:D-1){
      sdiff[s, d] ~ normal(0., 5); // sample specific bias
    }
  }

  for (r in 1:R){
    for (d in 1:D-1){
      bdiff[r, d] ~ normal(0., 5); // batch specific bias
    }
  }

  // generating counts
  for (m in 1:M){
    for (i in 1:D){
      target += neg_binomial_2_log_lpmf(y[m, i] | depth[m] + lam_clr[m, i], disp);
    }
  }
}
