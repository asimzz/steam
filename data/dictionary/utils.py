import random
from opencc import OpenCC
from itertools import permutations

# set seed
random.seed(0)

S2T = OpenCC('s2t')
T2S = OpenCC('t2s')

def transform(token, append_meta_symbols):
    def capitalize(s):
        return s.capitalize()

    def s2t(s):
        return S2T.convert(s)

    def t2s(s):
        return T2S.convert(s)

    def add_meta_symbols(s):
        # TODO: currently we only consider the sentencepiece meta symbol (U+2581)
        return f"▁{s}"

    # all permutations of the transformations
    transformations = []

    if append_meta_symbols:
        for r in range(1, 5):
            transformations.extend(permutations([capitalize, s2t, t2s, add_meta_symbols], r))
    else:
        for r in range(1, 4):
            transformations.extend(permutations([capitalize, s2t, t2s], r))

    res = [token]
    for t in transformations:
        new_token = token
        for f in t:
            new_token = f(new_token)
        res.append(new_token)

    # deduplicate
    res = list(set(res))
    return res