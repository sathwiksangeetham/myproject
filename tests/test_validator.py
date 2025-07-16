import sys
from types import ModuleType
from unittest.mock import MagicMock

# Mock external dependencies required when importing app.py
missing_modules = [
    'streamlit',
    'PyPDF2',
    'docx',
    'ollama',
    'plotly',
    'plotly.graph_objects',
    'plotly.express',
    'qrcode',
    'torch',
    'transformers',
    'nest_asyncio',
]

for name in missing_modules:
    if name not in sys.modules:
        module = ModuleType(name)
        sys.modules[name] = module

# Set up attributes for mocked modules used during import
sys.modules['plotly'].graph_objects = MagicMock()
sys.modules['plotly'].express = MagicMock()
sys.modules['transformers'].AutoTokenizer = MagicMock()
sys.modules['transformers'].AutoModelForCausalLM = MagicMock()
sys.modules['docx'].Document = MagicMock()
# Provide torch.classes.__path__ for compatibility
class _Classes:
    __path__ = []

sys.modules['torch'].classes = _Classes()
sys.modules['nest_asyncio'].apply = MagicMock()

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "app_module",
    Path(__file__).resolve().parents[1] / "app.py",
)
test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test)


def test_detect_keyword_stuffing_normal_text_false():
    validator = test.SmartValidator()
    text = " ".join(f"word{i}" for i in range(60))
    assert validator._detect_keyword_stuffing(text) is False


def test_detect_keyword_stuffing_repeated_keywords_true():
    validator = test.SmartValidator()
    text = "keyword " * 100
    assert validator._detect_keyword_stuffing(text) is True
