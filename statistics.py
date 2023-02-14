import pickle
import itertools
import numpy as np
from tqdm import tqdm
from collections import Counter

with open('train_tokenized_w_chunks.pt', mode="rb") as cache_f:
    train_data = pickle.load(cache_f)
with open('dev_tokenized_w_chunks.pt', mode="rb") as cache_f:
    dev_data = pickle.load(cache_f)
with open('test_tokenized_w_chunks.pt', mode="rb") as cache_f:
    test_data = pickle.load(cache_f)
all_data = []
all_data.extend(train_data)
all_data.extend(dev_data)
all_data.extend(test_data)

count_nested = 0
count_match_conll = 0
count_match_oia = 0
total_bounds = 0
count_seq_len = Counter()
count_span_len = Counter()
for item in tqdm(all_data, total=len(all_data)):
    count_seq_len[f'{len(item["cased_words"])}'] += 1
    for span in itertools.chain.from_iterable(item['span_clusters']):
        count_span_len[f'{span[1]-span[0]}'] += 1
    coref_boundaries = list(itertools.chain.from_iterable(itertools.chain.from_iterable(item['span_clusters'])))
    coref_boundaries = np.array(coref_boundaries)
    total_bounds += len(coref_boundaries)
    for i, v in enumerate(np.unique(np.unique(coref_boundaries, return_counts=True)[1], return_counts=True)[1]):
        if i >0:
            count_nested += v
    coref_bound_match_conll = np.zeros_like(coref_boundaries)
    coref_bound_match_oia = coref_bound_match_conll.copy()

    prev = 1
    for i, bound in enumerate(item['conll_bound']):
        if (prev == 1 and bound == 0) or bound == 1:
            # span start                 # span end
            if i in coref_boundaries:
                temp = np.where(coref_boundaries==i)
                for match_idx in temp[0]:
                    coref_bound_match_conll[match_idx] = 1
        prev = bound
        prev = 1

    for i, bound in enumerate(item['oia_bound']):
        if (prev == 1 and bound == 0) or bound == 1:
            # span start                 # span end
            if i in coref_boundaries:
                temp = np.where(coref_boundaries==i)
                for match_idx in temp[0]:
                    coref_bound_match_oia[match_idx] = 1
        prev = bound
    count_match_conll += np.count_nonzero(coref_bound_match_conll)
    count_match_oia += np.count_nonzero(coref_bound_match_oia)
    
    
print()