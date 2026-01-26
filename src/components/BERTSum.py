"""
PresummPackage

Lightweight wrapper class exposing common presumm functionality
for easier usage from other parts of the project.

This file does NOT duplicate module code; it imports the existing
modules under the `presumm` folder and provides simple methods that
call the corresponding functions. If your PYTHONPATH does not include
the presumm folder, adjust imports or run from project root.
"""
from typing import List, Dict, Any, Iterable
import importlib
import os
import sys

# Ensure package path is resolvable when imported from other places.
_HERE = os.path.dirname(__file__)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


class PresummPackage:
    """Wrapper for the presumm modules.

    Methods are thin adapters that call functions in the original
    modules. They keep a stable interface for external code.
    """

    def __init__(self):
        # lazy-import modules to avoid import-time side-effects
        self._modules = {}

    def _mod(self, name: str):
        if name in self._modules:
            return self._modules[name]
        # import from local presumm package
        module = importlib.import_module(f"presumm.{name}")
        self._modules[name] = module
        return module

    # cal_rouge wrapper
    def cal_rouge_process(self, candidates: List[str], references: List[str], pool_id: int = 0) -> Dict[str, Any]:
        """Call `process` from cal_rouge.py (creates temporary files and computes rouge).

        Returns the results dict (module may require pyrouge installed).
        """
        mod = self._mod('cal_rouge')
        # some versions expose `process` or `test_rouge` — prefer `process` if available
        if hasattr(mod, 'process'):
            return mod.process((candidates, references, pool_id))
        if hasattr(mod, 'test_rouge'):
            return mod.test_rouge(candidates, references, 1)
        raise AttributeError('cal_rouge module does not expose process/test_rouge')

    # preprocess wrappers
    def format_to_lines(self, args):
        mod = self._mod('preprocess')
        if hasattr(mod, 'do_format_to_lines'):
            return mod.do_format_to_lines(args)
        if hasattr(mod, 'data_builder') and hasattr(mod.data_builder, 'format_to_lines'):
            return mod.data_builder.format_to_lines(args)
        raise AttributeError('preprocess module lacks format_to_lines')

    def format_to_bert(self, args):
        mod = self._mod('preprocess')
        if hasattr(mod, 'do_format_to_bert'):
            return mod.do_format_to_bert(args)
        if hasattr(mod, 'data_builder') and hasattr(mod.data_builder, 'format_to_bert'):
            return mod.data_builder.format_to_bert(args)
        raise AttributeError('preprocess module lacks format_to_bert')

    def tokenize(self, args):
        mod = self._mod('preprocess')
        if hasattr(mod, 'do_tokenize'):
            return mod.do_tokenize(args)
        if hasattr(mod, 'data_builder') and hasattr(mod.data_builder, 'tokenize'):
            return mod.data_builder.tokenize(args)
        raise AttributeError('preprocess module lacks tokenize')

    # train wrappers — delegate to train_extractive / train_abstractive / train
    def train_extractive(self, args):
        mod = self._mod('train_extractive')

        # inference entry (test_text mode)
        if hasattr(mod, 'test_text_ext'):
            return mod.test_text_ext(args)

        # fallback: multi-gpu training
        if hasattr(mod, 'train_multi_ext'):
            return mod.train_multi_ext(args)
        raise AttributeError('train_extractive module lacks usable entry point')

    def train_abstractive(self, args):
        mod = self._mod('train_abstractive')
        if hasattr(mod, 'train_abs_multi'):
            return mod.train_abs_multi(args)
        if hasattr(mod, 'run'):
            return mod.run(args)
        raise AttributeError('train_abstractive module lacks entry point')

    def train_main(self, args):
        # general train entry from train.py
        mod = self._mod('train')
        if hasattr(mod, 'main'):
            return mod.main(args)
        if hasattr(mod, 'main_ori'):
            return mod.main_ori(args)
        raise AttributeError('train module lacks main entry')

    # post_stats wrappers
    def cal_self_repeat(self, summary: str):
        mod = self._mod('post_stats')
        if hasattr(mod, 'cal_self_repeat'):
            return mod.cal_self_repeat(summary)
        # fallback to local implementation
        return None

    def cal_novel(self, summary: str, gold: str, source: str, summary_ngram_novel: dict, gold_ngram_novel: dict):
        mod = self._mod('post_stats')
        if hasattr(mod, 'cal_novel'):
            return mod.cal_novel(summary, gold, source, summary_ngram_novel, gold_ngram_novel)
        return None


__all__ = ['PresummPackage']
