from biom import load_table
import numpy as np
import pandas as pd
import seaborn as sns
from patsy import dmatrix
import os

fname = '../data/File_1_TaoDing_microbiome.biom'
table = load_table(fname)

fname = 'File_3_FerretMicrobiomeMetadata.xlxs.xlsx'
fname = os.path.join('../data/', fname)
metadata = pd.read_excel(fname)

#id_counts = metadata.sampleID.value_counts()
#batches = id_counts.loc[id_counts>1].index
batch_metadata = metadata.set_index('sampleID')
batch_metadata = batch_metadata.reset_index()

# filter and match
batch_filter = lambda v, i, m: i in set(batch_metadata.SeqID)
read_filter = lambda v, i, m: v.sum()>100
obs_filter = lambda v, i, m: (v>0).sum()>5

table.filter(batch_filter, axis='sample')
table.filter(read_filter, axis='sample')
table.filter(obs_filter, axis='observation')

tab = table.to_dataframe().T.to_dense()
tab = tab.loc[batch_metadata.SeqID].dropna()
batch_metadata = batch_metadata.set_index('SeqID')
batch_metadata = batch_metadata.loc[tab.index]
batch_metadata = batch_metadata.reset_index()

# prepare ids for stan
subj_ids = batch_metadata.AnimalID.unique()
id_lookup = pd.Series(np.arange(len(subj_ids)), index=subj_ids)
per_sample_subj_ids = batch_metadata.AnimalID.apply(lambda x: id_lookup[x]).values + 1

run_ids = batch_metadata.runID.unique()
id_lookup = pd.Series(np.arange(len(run_ids)), index=run_ids)
per_sample_batch_ids = batch_metadata.runID.apply(lambda x: id_lookup[x]).values + 1

import pickle
pckl = pickle.load(open('../results/batch_results.pickle', 'rb'))
gamma = pckl['res']['bdiff'].mean(axis=0)
tab = tab.loc[:, pckl['biom'].columns]
batch_metadata['infected'] = batch_metadata['Days'] != 'uninfected'
batch_metadata.loc[batch_metadata['Days']=='unknown', 'infected'] = True
batch_metadata.loc[~batch_metadata['infected'], 'Days'] = 0
batch_metadata.loc[batch_metadata['Days']=='unknown', 'Days'] = 0
batch_metadata['Days'] = batch_metadata['Days'].astype(np.int64)

formula = 'Days + SampleType'
X = dmatrix(formula, batch_metadata, return_type='dataframe')

# Actual stan modeling
import pystan

code = open('../models/negative-binomial-regression.stan', 'r').read()
sm = pystan.StanModel(model_code=code)

dat = {
    'N' : X.shape[0],
    'R' : len(set(per_sample_batch_ids)),
    'D' : tab.shape[1],
    'p' : X.shape[1],
    'J' : len(set(per_sample_subj_ids)),
    'depth' : np.log(tab.sum(axis=1).values),
    'x' : X.values,
    'y' : tab.values.astype(np.int64),
    'subj_ids' : list(per_sample_subj_ids),
    'batch_ids' : list(per_sample_batch_ids),
    'gamma' : gamma
}

fit = sm.sampling(data=dat, iter=1000, chains=4)
res =  fit.extract(permuted=True)

# Save  model
pickle.dump({'dat': dat, 'res': res, 'biom': tab}, open('../results/nb_results_test.pickle', 'wb'))
