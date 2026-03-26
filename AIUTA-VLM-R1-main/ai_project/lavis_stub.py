"""
Stub out heavy lavis dependencies that VLFM never calls.
Usage: import lavis_stub  (before importing lavis)
"""
import sys, types

for mod in ['spacy', 'pycocoevalcap', 'pycocoevalcap.eval',
            'scikit-image', 'skimage', 'streamlit', 'plotly',
            'ipython', 'IPython', 'opendatasets', 'python_magic', 'magic']:
    sys.modules[mod] = types.ModuleType(mod)

sys.modules['spacy'].load = lambda *a, **k: None
sys.modules['pycocoevalcap.eval'].COCOEvalCap = type('COCOEvalCap', (), {})
