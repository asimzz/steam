#!/usr/bin/env python3
"""
Bayesian Optimization for back-translation language selection.

Uses BoTorch (SingleTaskGP + LogExpectedImprovement) to search over
linguistic feature vectors and find the back-translation language that
maximizes the γ_lang-corrected z-score.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood

import torch


logger = logging.getLogger(__name__)


def sample_initial_bt_langs(available_bt_langs: List[str], n_initial: int,
                            random_state: int, text_id: int) -> List[str]:
    """Sample n_initial random back-translation languages for a specific text."""
    rng = np.random.RandomState(random_state + text_id)
    n_select = min(n_initial, len(available_bt_langs))
    return rng.choice(available_bt_langs, n_select, replace=False).tolist()


def bo_suggest_next_bt_lang(evaluations: List[Dict[str, Any]],
                            available_bt_langs: List[str],
                            feature_vectors: Dict[str, np.ndarray]) -> Optional[str]:
    """
    Use BoTorch GP + LogEI to suggest next back-translation language.

    Evaluates the acquisition function directly at all unevaluated language
    feature vectors and picks the one with highest Expected Improvement.
    """
    evaluated_langs = {e['bt_lang'] for e in evaluations}
    remaining = [lang for lang in available_bt_langs if lang not in evaluated_langs]

    if not remaining:
        return None

    # Need at least 2 successful evaluations to fit a GP
    successful = [e for e in evaluations if e['success']]
    if len(successful) < 2:
        return np.random.choice(remaining)

    try:
        X_samples = [list(e['feature_vector']) for e in successful]
        y_samples = [e['z_score'] for e in successful]

        train_X = torch.from_numpy(np.array(X_samples)).double()
        train_Y = torch.tensor(y_samples, dtype=torch.double).unsqueeze(-1)

        # Fit GP surrogate model
        gp = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # Build tensor of ALL unevaluated candidate feature vectors
        candidate_features = torch.tensor(
            [feature_vectors[lang].tolist() for lang in remaining],
            dtype=torch.double
        )

        # Evaluate acquisition function at each discrete candidate
        best_f = train_Y.max()
        ei = LogExpectedImprovement(gp, best_f=best_f)
        # LogEI expects shape (batch, q, d) — add q=1 dimension
        ei_values = ei(candidate_features.unsqueeze(1))
        best_idx = ei_values.argmax().item()

        return remaining[best_idx]

    except Exception as e:
        logger.error(f"BO suggestion failed: {e}")
        return np.random.choice(remaining)


def optimize_single_text(text: str, prompt: str, text_id: int,
                         available_bt_langs: List[str],
                         feature_vectors: Dict[str, np.ndarray],
                         evaluate_fn, n_initial: int,
                         max_evaluations: int, random_state: int) -> Dict[str, Any]:
    """
    Run BO optimization for a single text.

    Args:
        text: The text to optimize back-translation language for
        prompt: The original prompt
        text_id: Index of the text (used for seeding)
        available_bt_langs: List of available back-translation languages (ISO-3)
        feature_vectors: Pre-computed feature vectors per language
        evaluate_fn: Callable(text, bt_lang) -> Dict with z_score, success, etc.
        n_initial: Number of initial random back-translation languages
        max_evaluations: Maximum total evaluations
        random_state: Base random seed

    Returns:
        Dict with {z_score, best_bt_lang_iso3, best_translated_text, prompt}
    """
    logger.info(f"Starting BO for text {text_id}")

    evaluations = []

    # Phase 1: Evaluate initial back-translation languages
    initial_bt_langs = sample_initial_bt_langs(available_bt_langs, n_initial, random_state, text_id)
    logger.info(f"  Initial back-translation languages for text {text_id}: {initial_bt_langs}")

    for bt_lang in initial_bt_langs:
        eval_result = evaluate_fn(text, bt_lang)
        evaluations.append(eval_result)
        logger.info(f"  Initial {bt_lang}: z={eval_result['z_score']:.3f}")

    # Find current best
    successful_evals = [e for e in evaluations if e['success']]
    if not successful_evals:
        logger.error(f"No successful evaluations for text {text_id}")
        return {'z_score': 0.0, 'best_bt_lang_iso3': None, 'best_translated_text': text, 'prompt': prompt}

    best_eval = max(successful_evals, key=lambda x: x['z_score'])

    # Phase 2: BO optimization loop
    for iteration in range(max_evaluations - n_initial):
        next_bt_lang = bo_suggest_next_bt_lang(evaluations, available_bt_langs, feature_vectors)
        if not next_bt_lang:
            break

        logger.info(f"  BO iteration {iteration + 1}: trying {next_bt_lang}")

        eval_result = evaluate_fn(text, next_bt_lang)
        evaluations.append(eval_result)

        if eval_result['success'] and eval_result['z_score'] > best_eval['z_score']:
            best_eval = eval_result

        logger.info(f"    {next_bt_lang}: z={eval_result['z_score']:.3f}")

    logger.info(f"  Text {text_id}: best={best_eval['bt_lang']}, z={best_eval['z_score']:.3f}")

    return {
        'z_score': best_eval['z_score'],
        'best_bt_lang_iso3': best_eval['bt_lang'],
        'best_translated_text': best_eval['translated_text'],
        'prompt': prompt,
    }
