"""
Language feature vectors from URIEL via lang2vec.

Provides 131-D vectors (syntax_knn + phonology_knn) for Bayesian Optimization.
"""

import lang2vec.lang2vec as l2v
import numpy as np


class LanguageFeatures:
    """Retrieve and cache language feature vectors from URIEL."""

    def __init__(self, feature_sets=None):
        if feature_sets is None:
            feature_sets = ['syntax_knn', 'phonology_knn']
        self.feature_sets = feature_sets
        self._cache = {}
        self.available_languages = set(l2v.available_languages())

    def get_feature_vector(self, lang):
        """Get concatenated feature vector for a language (ISO 639-3 code)."""
        if lang not in self._cache:
            vectors = []
            for fs in self.feature_sets:
                features = l2v.get_features([lang], fs)
                vectors.append(np.array(features[lang]))
            self._cache[lang] = np.concatenate(vectors)
        return self._cache[lang]
