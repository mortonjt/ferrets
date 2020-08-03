import pickle
import pandas

dres = pickle.load(open('../results/nb_results_test.pickle', 'rb'))
params = dres['dat']
res = dres['res']
print(type(res))
print(res.keys())
# res.to_csv('../results/nb_posterior.csv')
items = []
handle = open('../results/nb_100_summary.pickle', 'wb')
for k, v in res.items():
    print(k, v.shape)
    items.append((k, v[-100:].mean(axis=0), v[-100:].std(axis=0)))
pickle.dump(items, handle)

table = dres['biom']
table.to_csv('../results/table.csv')
