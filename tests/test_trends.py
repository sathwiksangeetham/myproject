import types
import sys

# Stub out heavy modules before importing test.py
st = types.ModuleType('streamlit')
st.session_state = types.SimpleNamespace(settings={})
sys.modules['streamlit'] = st
for mod_name in [
    'PyPDF2', 'docx', 'ollama', 'plotly', 'plotly.graph_objects',
    'plotly.express', 'torch', 'transformers', 'qrcode', 'nest_asyncio']:
    sys.modules[mod_name] = types.ModuleType(mod_name)

sys.modules['docx'].Document = lambda *a, **k: None
sys.modules['transformers'].AutoTokenizer = object
sys.modules['transformers'].AutoModelForCausalLM = object
sys.modules['torch'].classes = types.SimpleNamespace(__path__=[])
sys.modules['nest_asyncio'].apply = lambda: None
sys.modules['plotly.graph_objects'].Figure = object
sys.modules['plotly.express'].bar = lambda *a, **k: None

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "project_test", Path(__file__).resolve().parents[1] / "test.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

import pandas as pd
from datetime import datetime


def test_trend_improving():
    engine = module.AnalyticsEngine()
    df = pd.DataFrame(
        {
            'date': [datetime(2024, 1, 1), datetime(2024, 1, 8)],
            'match_score': [50, 80],
        }
    )
    result = engine._calculate_trends(df)
    assert result['direction'] == 'improving'
    assert result['change'] > 0


def test_trend_declining():
    engine = module.AnalyticsEngine()
    df = pd.DataFrame(
        {
            'date': [datetime(2024, 1, 1), datetime(2024, 1, 8)],
            'match_score': [80, 50],
        }
    )
    result = engine._calculate_trends(df)
    assert result['direction'] == 'declining'
    assert result['change'] < 0


def test_trend_stable():
    engine = module.AnalyticsEngine()
    df = pd.DataFrame(
        {
            'date': [datetime(2024, 1, 1), datetime(2024, 1, 8)],
            'match_score': [70, 70],
        }
    )
    result = engine._calculate_trends(df)
    assert result['direction'] == 'stable'
    assert result['change'] == 0
