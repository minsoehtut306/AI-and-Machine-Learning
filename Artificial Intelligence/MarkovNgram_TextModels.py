"""
Markov n-gram text models (unigram, bigram, general n-gram) with:
- Model builders
- Context queries
- Blended predictions (mixture over models)
- Next-token prediction
- Log-likelihood (ramp-up and blended)

Designed to work directly with a plain-text corpus file.
"""

import math
import random
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------
# Tokenization
# ---------------------------

def tokenise(filename: str) -> List[str]:
    """
    Load a text file and split into tokens.

    - Lowercases
    - Replaces underscores with spaces
    - Splits on digits or non-word characters, *keeping* single char tokens
    - Removes whitespace-only tokens and newlines
    """
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read().replace("_", " ").lower()
    # Split on (digit OR non-word), capture delimiters, then filter empties/spaces/newlines.
    tokens = [t for t in re.split(r"(\d|\W)", text) if t and t not in (" ", "\n")]
    return tokens


# ---------------------------
# Model builders
# ---------------------------

Model = Dict[Tuple[str, ...], Dict[str, int]]

def build_unigram(sequence: Sequence[str]) -> Model:
    """
    Build a unigram model: {(): {token: count, ...}}
    Context is the empty tuple ().
    """
    counts: Dict[str, int] = {}
    for tok in sequence:
        counts[tok] = counts.get(tok, 0) + 1
    return {(): counts}


def build_bigram(sequence: Sequence[str]) -> Model:
    """
    Build a bigram model:
      {(w_{i-1},): {w_i: count, ...}, ...}
    """
    model: Model = {}
    for i in range(len(sequence) - 1):
        ctx = (sequence[i],)
        nxt = sequence[i + 1]
        bucket = model.setdefault(ctx, {})
        bucket[nxt] = bucket.get(nxt, 0) + 1
    return model


def build_n_gram(sequence: Sequence[str], n: int) -> Model:
    """
    General n-gram model:
      { (w_{i-n+1},...,w_{i-1}): { w_i: count, ... } }

    n=1 reduces to unigram: context = ()
    n=2 is bigram, etc.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    model: Model = {}
    if n == 1:
        return build_unigram(sequence)
    for i in range(len(sequence) - n + 1):
        ctx = tuple(sequence[i : i + n - 1])
        nxt = sequence[i + n - 1]
        bucket = model.setdefault(ctx, {})
        bucket[nxt] = bucket.get(nxt, 0) + 1
    return model


# ---------------------------
# Model queries & utilities
# ---------------------------

def query_n_gram(model: Model, context: Sequence[str]) -> Optional[Dict[str, int]]:
    """
    Return the dictionary of continuation counts given a context.

    - For unigram models, pass context=() (empty tuple). You can also pass any
      sequence; it will be turned into a tuple and looked up (only () exists).
    - For n>=2, return counts if the exact context exists; else None.
    """
    return model.get(tuple(context))


def _normalize(counts: Dict[str, int]) -> Dict[str, float]:
    """Convert {token: count} to {token: prob} (sums to 1)."""
    total = float(sum(counts.values()))
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def _blend_dicts(distributions: List[Dict[str, float]], weights: List[float]) -> Dict[str, float]:
    """
    Weighted sum of probability distributions over the same token space.
    Missing tokens treated as 0. Weights are assumed to sum to 1.
    """
    out: Dict[str, float] = {}
    for dist, w in zip(distributions, weights):
        if w <= 0:
            continue
        for tok, p in dist.items():
            out[tok] = out.get(tok, 0.0) + w * p
    # Re-normalize to correct any tiny FP drift
    s = sum(out.values())
    if s > 0:
        for tok in list(out.keys()):
            out[tok] /= s
    return out


# ---------------------------
# Blending predictions
# ---------------------------

def blend_predictions(preds: List[Optional[Dict[str, int]]], factor: float = 0.8) -> Dict[str, float]:
    """
    Blend predictions from multiple models.

    Steps:
      1) Drop None predictions
      2) Normalize each to a probability distribution
      3) Apply geometric weighting:
         w0 = factor
         w1 = (1-factor) * factor
         ...
         w_{k-1} = (1-factor) ** (k-1) * factor
         The *last* prediction gets all remaining weight so weights sum to 1.

    Returns:
      dict {token: blended_prob}, sums to 1. Empty dict if nothing to blend.
    """
    filtered = [p for p in preds if p is not None]
    if not filtered:
        return {}

    normed = [_normalize(p) for p in filtered]
    # Compute geometric weights that sum to 1
    weights: List[float] = []
    remaining = 1.0
    for i in range(len(normed)):
        if i < len(normed) - 1:
            w = factor * remaining
            remaining -= w
        else:
            w = remaining  # last takes all remaining
        weights.append(w)
    return _blend_dicts(normed, weights)


# ---------------------------
# Next-token prediction
# ---------------------------

def predict(history: Sequence[str], models: List[Model], factor: float = 0.8) -> Optional[str]:
    """
    Sample a next token from the blended probability across all *applicable* models.

    For each model:
      - Let m = len(any_key)  (the key length is n-1)
      - If len(history) >= m, use the last m tokens as context
      - Query model; collect predictions; blend; sample.

    Returns a sampled token, or None if no model had predictions.
    """
    preds: List[Optional[Dict[str, int]]] = []
    for model in models:
        # Every key in a given model has same length (context size)
        any_key = next(iter(model.keys()))
        m = len(any_key)  # context length = n-1
        ctx = tuple(history[-m:]) if m > 0 else ()
        preds.append(query_n_gram(model, ctx))

    blended = blend_predictions(preds, factor=factor)
    if not blended:
        return None

    tokens, probs = zip(*blended.items())
    return random.choices(list(tokens), weights=list(probs), k=1)[0]


# ---------------------------
# Log-likelihoods
# ---------------------------

def _context_size(model: Model) -> int:
    """Return context length (n-1) from any key of the model."""
    return len(next(iter(model.keys())))


def log_likelihood_ramp_up(sequence: Sequence[str], models: List[Model]) -> float:
    """
    Chain-rule log-likelihood with a 'ramp-up' of model orders.

    Assumes `models` are ordered from highest n to lowest (ending with unigram),
    e.g., [5-gram, 4-gram, 3-gram, 2-gram, 1-gram].
    The first token uses unigram, second uses bigram, ... ramping up
    until we reach the highest-order model; thereafter we stick with it.

    If any required context->token pair is unseen (probability 0), returns -inf.
    """
    if not sequence:
        return 0.0

    # Defensive copy: we want lowest to highest for ramp-up logic
    models_desc = list(models)             # given order: assume high->low
    models_asc  = list(reversed(models))   # low->high for ramp-up

    total_logp = 0.0

    for i, tok in enumerate(sequence):
        # Choose which model to use at position i:
        # 0 -> unigram (index 0), 1 -> bigram (index 1), ..., then clamp to highest.
        idx = min(i, len(models_asc) - 1)
        model = models_asc[idx]
        m = _context_size(model)
        ctx = tuple(sequence[i - m:i]) if m > 0 else ()

        counts = query_n_gram(model, ctx)
        if not counts:
            return -math.inf
        total = sum(counts.values())
        prob = counts.get(tok, 0) / total if total > 0 else 0.0
        if prob <= 0.0:
            return -math.inf
        total_logp += math.log(prob)

    return total_logp


def log_likelihood_blended(sequence: Sequence[str], models: List[Model], factor: float = 0.8) -> float:
    """
    Chain-rule log-likelihood using *blended* model probabilities at each step.

    At position i, gather predictions from all models whose context fits,
    blend them, read off P(token_i | context), accumulate log-probability.
    If any step yields zero probability for the actual token, returns -inf.
    """
    if not sequence:
        return 0.0

    total_logp = 0.0
    for i, tok in enumerate(sequence):
        preds: List[Optional[Dict[str, int]]] = []

        for model in models:
            m = _context_size(model)
            if i >= m:
                ctx = tuple(sequence[i - m:i]) if m > 0 else ()
                preds.append(query_n_gram(model, ctx))
            else:
                preds.append(None)

        blended = blend_predictions(preds, factor=factor)
        prob = blended.get(tok, 0.0)
        if prob <= 0.0:
            return -math.inf
        total_logp += math.log(prob)

    return total_logp


# ---------------------------
# Demo / quick test
# ---------------------------

if __name__ == "__main__":
    # Change this filename if you rename your corpus file.
    CORPUS = "MarkovNgram_corpus.txt"

    seq = tokenise(CORPUS)

    # Build a family of models (highest to lowest n). Tweak the range if desired.
    # Example here: 6-gram down to unigram.
    models_high_to_low = [build_n_gram(seq, n) for n in range(6, 0, -1)]

    # Generate a short sample
    history: List[str] = []
    print("Sample generation:")
    for _ in range(40):
        nxt = predict(history, models_high_to_low, factor=0.8)
        if nxt is None:
            break
        print(nxt, end=" ")
        history.append(nxt)
    print("\n")

    # Small likelihood sanity checks on a short slice
    short = seq[:30]
    print("log_likelihood_ramp_up(short):", log_likelihood_ramp_up(short, models_high_to_low))
    print("log_likelihood_blended(short):", log_likelihood_blended(short, models_high_to_low, factor=0.8))
