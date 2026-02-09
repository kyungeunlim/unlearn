import numpy as np


def process_results(doc, results):
    """Return the softmax probability of the correct answer."""
    lls, is_greedy = zip(*results)
    lls = np.array(lls)

    gold = doc["answer"]
    if isinstance(gold, str):
        gold = doc["choices"].index(gold)

    # Softmax with log-sum-exp for numerical stability
    max_ll = np.max(lls)
    probs = np.exp(lls - max_ll)
    probs = probs / np.sum(probs)

    return {
        "correct_prob": float(probs[gold]),
    }
