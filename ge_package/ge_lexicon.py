"""Paths and the assembly lexicon shared by ge_terms.py, ge_naming_audit.py,
ge_tfidf_masked.py, ge_queue_contrasts.py and ge_contrasts.py. Their output
goes to results_contrasts/ (or GE_RES_CONTRASTS), so the runs under
results/ are only read. Word lists drawn from the records go to private/
and are not exported.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("GE_RES_CONTRASTS", os.path.join(HERE, "results_contrasts"))
PRIVATE = os.environ.get("GE_PRIVATE", os.path.join(HERE, "private"))
MASK_DIR = os.environ.get("GE_MASKS", os.path.join(HERE, "masks"))
LEXICON = os.path.join(HERE, "assembly_lexicon.json")
# assembly names, no-fault terms and replacement verbs
MASK_FULL = os.path.join(MASK_DIR, "outcome_terms_v2.txt")
# assembly names only
MASK_NAMES = os.path.join(MASK_DIR, "assembly_names_v2.txt")
FIELDS = ("customer", "technician", "repair")


def load_lexicon():
    with open(LEXICON) as f:
        lexicon = json.load(f)
    assemblies = {}
    for name, terms in lexicon["assemblies"].items():
        assemblies[name] = {term.lower() for term in terms}
    no_fault = {term.lower() for term in lexicon["no_fault"]}
    verbs = {term.lower() for term in lexicon["replacement_verbs"]}
    class_parts = {int(label): set(parts) for label, parts in lexicon["class_parts"].items()}
    return assemblies, no_fault, verbs, class_parts
