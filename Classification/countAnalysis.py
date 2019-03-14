#%%
import os
import numpy as np
import pandas as pd
from functools import partial
from nltk.tokenize.regexp import regexp_tokenize
from sklearn.feature_extraction.text import CountVectorizer

#%%
GOOD_PATH = r"D:\Documents\Programming\ML\DeepLearning-add-3-models\Data\subsets\benign_badging_1000h.txt"
BAD_PATH = r"D:\Documents\Programming\ML\DeepLearning-add-3-models\Data\subsets\mal_badging_1000h.txt"

with open(GOOD_PATH, encoding='utf-8') as f:
    ben_samples = f.readlines()
with open(BAD_PATH, encoding='utf-8') as f:
    mal_samples = f.readlines()

#%%
samples = ben_samples + mal_samples

#%%
perm_pattern = "(?:\w|\.)+(?:permission).(?:\w|\.)+"
feat_pattern = "(?:\w|\.)+(?:hardware).(?:\w|\.)+"
comb_pattern = "(?:\w|\.)+(?:hardware|permission).(?:\w|\.)+"

perm_vect = CountVectorizer(analyzer=partial(
    regexp_tokenize, pattern=perm_pattern))
feat_vect = CountVectorizer(analyzer=partial(
    regexp_tokenize, pattern=feat_pattern))
comb_vect = CountVectorizer(analyzer=partial(
    regexp_tokenize, pattern=comb_pattern))

X = comb_vect.fit_transform(samples)

#%%
X_arr = X.toarray()
X_sum = np.sum(X_arr, axis=0)
X_sum_sorted = np.argsort(X_sum)

#%%
X_sum_sorted

#%%
X_cutoff = X_sum_sorted[len(X_sum_sorted):len(X_sum_sorted)-50:-1]
X_cutoff

#%%
def importantVocabCount(row):
    sum = 0
    for i in X_cutoff:
        sum += row[i]
    return sum

X_importantWordCount = np.apply_along_axis(importantVocabCount, axis=1, arr=X_arr)
X_importantWordCount

#%%
X_importantWordCountSorted = np.argsort(X_importantWordCount.ravel())
X_importantWordCountSorted.shape

#%%
X_importantWordCountSorted[::-1]

#%%
X_importantWordCountSorted[2000:1988:-1]


