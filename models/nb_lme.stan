data {
  int<lower=0> N;                // number of samples
  int<lower=0> R;                // number of runs (batches)
  int<lower=0> D;                // number of microbes
  int<lower=0> J;                // number of subjects
  int<lower=0> p;                // number of sample covariates
  int<lower=0> q;                // number of batch covariates
  real depth[N];                 // sequencing depths of microbes
  matrix[N, p] x;                // covariate matrix

  int<lower=1, upper=J> subj_ids[N];   // subject ids
  int<lower=1, upper=J> batch_ids[N];  // batch ids
  int y[N, D];                         // observed microbe abundances
}

parameters {
  // parameters required for linear regression on the species means
  matrix[p, D-1] beta;                 // covariate differnetials
  matrix[J, D-1] alpha;                // subject differentials
  matrix[R, D-1] gamma;                // batch differnetials
  real reciprocal_phi;
}

transformed parameters {
  matrix[N, D-1] lam;
  matrix[N, D] lam_clr;
  matrix[N, D] prob;
  vector[N] z;
  real phi;

  phi = 1. / reciprocal_phi;


  z = to_vector(rep_array(0, N));
  lam = x * beta;
  // add batch effects
  for (m in 1:M){
    lam[samp_ids[m]] += gamma[batch_ids[m]]
  }
  // add in subject specific effects
  for (m in 1:M){
    lam[samp_ids[m]] += gamma[batch_ids[m]]
  }


  lam_clr = append_col(z, lam);

}

model {



  // setting priors ...
  reciprocal_phi ~ cauchy(0., 5.);
  vec(beta) ~ normal(0., 5.); // covariate differential
  vec(alpha) ~ normal(0., 5.); // subject specific bias
  vec(gamma) ~ normal(0., 5.); // batch specific bias

  // generating counts
  for (n in 1:N){
    for (i in 1:D){
      target += neg_binomial_2_log_lpmf(y[n, i] | depth[n] + lam_clr[n, i], phi);
    }
  }
}
