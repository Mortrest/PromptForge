import nltk
from nltk.corpus import wordnet as wn
import random

# Ensure WordNet is downloaded
try:
    wn.synsets('test')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def get_synonym(word):
    """
    Return a random synonym for the word using WordNet, or the original word if no synonym is found.
    """
    synsets = wn.synsets(word)
    lemmas = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            if lemma.name().lower() != word.lower():
                lemmas.add(lemma.name().replace('_', ' '))
    if lemmas:
        return random.choice(list(lemmas))
    return word

def synonym_perturb_kg(triples, percent=100):
    """
    Given a list of (subject, relation, object) triples, replace nodes with synonyms in N% of the triples.
    percent: integer from 0 to 100, percent of triples to perturb.
    """
    n = len(triples)
    k = int(n * percent / 100)
    indices = set(random.sample(range(n), k)) if k > 0 else set()
    perturbed = []
    for i, (s, r, o) in enumerate(triples):
        if i in indices:
            s_syn = get_synonym(s)
            o_syn = get_synonym(o)
            perturbed.append((s_syn, r, o_syn))
        else:
            perturbed.append((s, r, o))
    return perturbed

