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

id_counts = metadata.sampleID.value_counts()
batches = id_counts.loc[id_counts>1].index
batch_metadata = metadata.set_index('sampleID').loc[batches]
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
sample_ids = batch_metadata.sampleID.unique()
id_lookup = pd.Series(np.arange(len(sample_ids)), index=sample_ids)
per_sample_ids = batch_metadata.sampleID.apply(lambda x: id_lookup[x]).values + 1

run_ids = batch_metadata.runID.unique()
id_lookup = pd.Series(np.arange(len(run_ids)), index=run_ids)
per_sample_batch_ids = batch_metadata.runID.apply(lambda x: id_lookup[x]).values + 1


# Actual stan modeling
import pystan

code = open('../models/nb_batch.stan', 'r').read()
sm = pystan.StanModel(model_code=code)


dat = {
    'M' : batch_metadata.shape[0],
    'R' : len(set(per_sample_batch_ids)),
    'D' : table.shape[0],
    'S' : len(set(per_sample_ids)),
    'depth' : np.log(tab.sum(axis=1)).values,
    'samp_ids' : per_sample_ids,
    'batch_ids' : per_sample_batch_ids,
    'y' : tab.values.astype(np.int)
}

fit = sm.sampling(data=dat, iter=1000, chains=4)
res =  fit.extract(permuted=True)

# Save mdoel
import pickle

pickle.dump({'dat': dat, 'res': res, 'biom': tab}, open('../results/batch_results.pickle', 'wb'))
