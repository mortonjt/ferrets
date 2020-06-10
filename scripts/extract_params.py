import pandas as pd
import pickle
import glob
import sys
import re

s = r'posterior_(\d+).pickle'
pattern = re.compile(s)
# ../results/stan_params/
posterior_dir = sys.argv[1]
results_dir = sys.argv[2]
files = glob.glob(f'{posterior_dir}/*.pickle')
results = []
index = []
alpha_means = []
alpha_stds = []
beta_means = []
beta_stds = []

for f in files:
    i = int(pattern.findall(f)[0])
    dump = pickle.load(open(f, 'rb'))
    taxa_name = dump['microbes'][i]
    res = dump['res']
    beta_mean = res['beta'][-100:].mean(axis=0)
    beta_std = res['beta'][-100:].std(axis=0)
    alpha_mean = res['alpha'][-100:].mean(axis=0)
    alpha_std = res['alpha'][-100:].std(axis=0)
    index.append(taxa_name)
    alpha_means.append(alpha_mean)
    alpha_stds.append(alpha_std)
    beta_means.append(beta_mean)
    beta_stds.append(beta_std)

out_files = ['alpha_means', 'alpha_stds', 'beta_means', 'beta_stds']
for f in out_files:
    outf = f'{results_dir}/{f}.csv'
    if 'beta' in f:
        df = pd.DataFrame(eval(f), index=index, columns=['Intercept', 'Days'])
        df.to_csv(outf)
    else:
        df = pd.DataFrame(eval(f), index=index)
        df.to_csv(outf)
