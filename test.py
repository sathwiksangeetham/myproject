import streamlit as st
import json
import os
from datetime import datetime, timedelta
import uuid
import logging
import time
from pathlib import Path
import re
from typing import Dict, List, Optional, Any, Tuple, Iterator
import PyPDF2
from docx import Document
import ollama
from dataclasses import dataclass, asdict, field
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
from functools import lru_cache
import hashlib
import pickle
import secrets
import qrcode
from io import BytesIO
import base64
import html
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import threading
from queue import Queue
from enum import Enum

# ====================================================================================
# FIX 1: Suppress TensorFlow and PyTorch warnings
# ====================================================================================

import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress general warnings
warnings.filterwarnings('ignore')

# Fix for PyTorch/Streamlit compatibility
import sys
if hasattr(sys, 'modules'):
    if 'torch' in sys.modules:
        import torch
        # Disable torch classes path inspection
        torch.classes.__path__ = []

# ====================================================================================
# FIX 3: Add async compatibility wrapper
# ====================================================================================

import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

def run_async(coro):
    """Run async function in Streamlit context"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)

# ====================================================================================
# FIX 6: Add requirements check at startup
# ====================================================================================

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'plotly': 'plotly',
        'PyPDF2': 'PyPDF2',
        'docx': 'python-docx',
        'transformers': 'transformers',
        'torch': 'torch',
        'nest_asyncio': 'nest-asyncio'
    }
    
    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        st.error(f"Missing required packages: {', '.join(missing_packages)}")
        st.code(f"pip install {' '.join(missing_packages)}")
        st.stop()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path("data")
JOBS_DIR = DATA_DIR / "jobs"
RESUMES_DIR = DATA_DIR / "resumes"
COMPARISONS_DIR = DATA_DIR / "comparisons"
RESUME_FILES_DIR = DATA_DIR / "resume_files"
CACHE_DIR = DATA_DIR / "cache"
SETTINGS_FILE = DATA_DIR / "settings.json"

# Difficulty-based constants for faster calculations
DIFFICULTY_WEEKS = {
    "Hard": 8,  # 2 months for hard skills
}
DEFAULT_WEEKS_PER_SKILL = 3
DIFFICULTY_PROGRESS = {
    "Hard": 0.3,
}
DEFAULT_PROGRESS = 0.6

# AI Processing State Management
class AIProcessingStage(Enum):
    INITIALIZING = "Initializing AI Model"
    ANALYZING_FORMAT = "Analyzing Document Format"
    EXTRACTING_STRUCTURE = "Extracting Document Structure"
    PARSING_CONTENT = "Parsing Content Fields"
    VALIDATING_DATA = "Validating Extracted Data"
    REFINING_RESULTS = "Refining Results"
    FINALIZING = "Finalizing Analysis"

@dataclass
class AIThought:
    stage: AIProcessingStage
    thought: str
    confidence: float
    tokens_generated: int
    processing_time: float
    current_field: str = ""
    
@dataclass
class StreamingProgress:
    current_stage: AIProcessingStage = AIProcessingStage.INITIALIZING
    total_stages: int = 7
    current_tokens: int = 0
    total_estimated_tokens: int = 1000
    thoughts: List[AIThought] = field(default_factory=list)
    partial_result: Dict = field(default_factory=dict)
    overall_confidence: float = 0.0
    processing_speed: float = 0.0  # tokens per second

# Smart Cache Implementation
class SmartCache:
    def __init__(self, cache_dir="data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}
    
    def get_cache_key(self, text: str, operation: str) -> str:
        """Generate unique cache key"""
        content_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{operation}_{content_hash}"
    
    def get(self, key: str, max_age_hours: int = 24):
        """Get from cache with age check"""
        # Memory cache first
        if key in self.memory_cache:
            self.cache_stats["hits"] += 1
            return self.memory_cache[key]
        
        # Disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if age < timedelta(hours=max_age_hours):
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.memory_cache[key] = data
                    self.cache_stats["hits"] += 1
                    return data
        
        self.cache_stats["misses"] += 1
        return None
    
    def set(self, key: str, value: Any):
        """Save to cache"""
        self.memory_cache[key] = value
        cache_file = self.cache_dir / f"{key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(value, f)
    
    @property
    def hit_rate(self):
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        return self.cache_stats["hits"] / total if total > 0 else 0
    
    def clear_cache(self):
        """Clear all cached data"""
        self.memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        self.cache_stats = {"hits": 0, "misses": 0}

# Initialize globally
cache = SmartCache()

# Cached wrapper functions
def cached_llm_parse(text: str, parse_type: str, model: str = "SmolLM2-135M", 
                    progress_callback=None, thought_callback=None) -> Dict:
    cache_key = cache.get_cache_key(text, f"parse_{parse_type}_{model}")
    
    # Check cache first
    cached_result = cache.get(cache_key)
    if cached_result:
        st.caption(f"⚡ Using cached result (Cache hit rate: {cache.hit_rate*100:.1f}%)")
        return cached_result
    
    # Parse with streaming
    result = None
    for partial_result, progress in llm_parse_streaming(
        text, parse_type, progress_callback, thought_callback
    ):
        result = partial_result
    
    # Cache successful results
    if result and "error" not in str(result.values()).lower():
        cache.set(cache_key, result)
    
    return result

async def cached_multi_model_parse(text: str, parse_type: str, models: List[str] = None) -> Tuple[Dict, float]:
    """Cached version of multi_model_parse"""
    cache_key = cache.get_cache_key(text, f"multi_parse_{parse_type}_{'_'.join(models or [])}")
    
    # Check cache first
    cached_result = cache.get(cache_key)
    if cached_result:
        st.caption(f"⚡ Using cached result (Cache hit rate: {cache.hit_rate*100:.1f}%)")
        return cached_result
    
    # Parse and cache
    result = await parser.multi_model_parse(text, parse_type, models)
    cache.set(cache_key, result)
    return result

# Batch processing for multiple files
async def batch_process_resumes(files: List) -> List[Dict]:
    """Process multiple resumes in parallel"""
    async def process_file(file):
        try:
            text = extract_text_from_file(file)
            return await cached_multi_model_parse(text, 'resume')
        except Exception as e:
            return None, str(e)
    
    tasks = [process_file(f) for f in files]
    results = await asyncio.gather(*tasks)
    return results

async def batch_process_jobs(job_texts: List[str]) -> List[Dict]:
    """Process multiple job descriptions in parallel"""
    async def process_job(text):
        try:
            return await cached_multi_model_parse(text, 'job')
        except Exception as e:
            return None, str(e)
    
    tasks = [process_job(text) for text in job_texts]
    results = await asyncio.gather(*tasks)
    return results

# Create directories
for dir_path in [DATA_DIR, JOBS_DIR, RESUMES_DIR, COMPARISONS_DIR, RESUME_FILES_DIR, CACHE_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data Models
@dataclass
class JobDescription:
    id: str
    title: str
    company: str
    location: str
    type: str
    responsibilities: List[str]
    required_skills: List[str]
    preferred_skills: List[str]
    education: str
    experience: str
    raw_text: str
    date_added: str

@dataclass
class Experience:
    title: str
    company: str
    duration: str
    responsibilities: List[str]

@dataclass
class Resume:
    id: str
    name: str
    email: str
    phone: str
    summary: str
    experience: List[Dict]
    skills: List[str]
    education: str
    raw_text: str

@dataclass
class ComparisonResult:
    match_score: float
    overall_summary: str
    skill_matches_detailed: Dict
    experience_match_status: str
    education_match_status: str
    ai_recommendations: List[str]
    missing_keywords: List[str]
    ai_confidence_scores: Dict

# Validation Module
@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_data: Optional[Dict]

class SmartValidator:
    def __init__(self):
        self.email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        self.phone_pattern = re.compile(r'^\+?1?\d{9,15}$')
        self.url_pattern = re.compile(r'^https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b')
        
    def validate_resume(self, resume_data: Dict) -> ValidationResult:
        """Comprehensive resume validation"""
        errors = []
        warnings = []
        
        # Get validation settings
        validation_settings = st.session_state.settings.get('validation', {})
        min_skills = validation_settings.get('min_skills', 3)
        max_skills = validation_settings.get('max_skills', 50)
        keyword_threshold = validation_settings.get('keyword_threshold', 2)
        strict_validation = validation_settings.get('strict_validation', False)
        auto_sanitize = validation_settings.get('auto_sanitize', True)
        
        # Required fields
        if not resume_data.get('name'):
            errors.append("Name is required")
        
        # Email validation
        email = resume_data.get('email', '')
        if email and not self.email_pattern.match(email):
            errors.append("Invalid email format")
        
        # Phone validation
        phone = resume_data.get('phone', '').replace(' ', '').replace('-', '')
        if phone and not self.phone_pattern.match(phone):
            warnings.append("Phone number format may be incorrect")
        
        # Skills validation
        skills = resume_data.get('skills', [])
        if len(skills) < min_skills:
            warnings.append(f"Consider adding more skills (minimum {min_skills} recommended)")
        if len(skills) > max_skills:
            warnings.append(f"Too many skills listed - consider focusing on top {max_skills//2}-{max_skills}")
        
        # Experience validation
        experience = resume_data.get('experience', [])
        if not experience:
            warnings.append("No work experience listed")
        
        # Detect potential issues
        if self._detect_keyword_stuffing(resume_data.get('raw_text', ''), keyword_threshold):
            warnings.append("Potential keyword stuffing detected - use keywords naturally")
        
        # Convert warnings to errors in strict mode
        if strict_validation:
            errors.extend(warnings)
            warnings = []
        
        # Sanitize data if enabled
        if auto_sanitize:
            sanitized = self._sanitize_resume_data(resume_data)
        else:
            sanitized = resume_data
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=sanitized
        )
    
    def validate_job(self, job_data: Dict) -> ValidationResult:
        """Comprehensive job description validation"""
        errors = []
        warnings = []
        
        # Get validation settings
        validation_settings = st.session_state.settings.get('validation', {})
        min_job_skills = validation_settings.get('min_job_skills', 2)
        max_job_skills = validation_settings.get('max_job_skills', 30)
        keyword_threshold = validation_settings.get('keyword_threshold', 2)
        strict_validation = validation_settings.get('strict_validation', False)
        auto_sanitize = validation_settings.get('auto_sanitize', True)
        
        # Required fields
        if not job_data.get('title'):
            errors.append("Job title is required")
        
        if not job_data.get('company'):
            warnings.append("Company name is recommended")
        
        # Skills validation
        required_skills = job_data.get('required_skills', [])
        if len(required_skills) < min_job_skills:
            warnings.append(f"Consider adding more required skills (minimum {min_job_skills})")
        if len(required_skills) > max_job_skills:
            warnings.append(f"Too many required skills - consider prioritizing top {max_job_skills//2}")
        
        # Experience validation
        experience = job_data.get('experience', '')
        if not experience:
            warnings.append("Experience requirements not specified")
        
        # Detect potential issues
        if self._detect_keyword_stuffing(job_data.get('raw_text', ''), keyword_threshold):
            warnings.append("Potential keyword stuffing detected in job description")
        
        # Convert warnings to errors in strict mode
        if strict_validation:
            errors.extend(warnings)
            warnings = []
        
        # Sanitize data if enabled
        if auto_sanitize:
            sanitized = self._sanitize_job_data(job_data)
        else:
            sanitized = job_data
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=sanitized
        )
    
    def _detect_keyword_stuffing(self, text: str, threshold: float = 2.0) -> bool:
        """Detect unnatural keyword repetition"""
        if not text:
            return False
            
        words = text.lower().split()
        word_count = {}
        
        for word in words:
            if len(word) > 3:  # Skip short words
                word_count[word] = word_count.get(word, 0) + 1
        
        # Check for suspicious repetition
        total_words = len(words)
        if total_words == 0:
            return False
            
        threshold_ratio = threshold / 100.0  # Convert percentage to ratio
        for word, count in word_count.items():
            if count / total_words > threshold_ratio:
                return True
        
        return False
    
    def _sanitize_resume_data(self, data: Dict) -> Dict:
        """Sanitize resume input data"""
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str):
                # Remove HTML tags and escape special characters
                value = re.sub(r'<[^>]+>', '', value)
                value = html.escape(value)
                sanitized[key] = value.strip()
            elif isinstance(value, list):
                sanitized[key] = [self._sanitize_string(item) if isinstance(item, str) else item 
                                 for item in value]
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _sanitize_job_data(self, data: Dict) -> Dict:
        """Sanitize job input data"""
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str):
                # Remove HTML tags and escape special characters
                value = re.sub(r'<[^>]+>', '', value)
                value = html.escape(value)
                sanitized[key] = value.strip()
            elif isinstance(value, list):
                sanitized[key] = [self._sanitize_string(item) if isinstance(item, str) else item 
                                 for item in value]
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _sanitize_string(self, text: str) -> str:
        """Sanitize a single string"""
        if not text:
            return ""
        # Remove HTML tags and escape special characters
        text = re.sub(r'<[^>]+>', '', text)
        text = html.escape(text)
        return text.strip()

# Initialize validator globally
validator = SmartValidator()

# Analytics Engine
class AnalyticsEngine:
    def generate_insights(self, comparisons: List[Dict]) -> Dict:
        """Generate comprehensive analytics"""
        if not comparisons:
            return {}
        
        df = pd.DataFrame(comparisons)
        df['date'] = pd.to_datetime(df['date'])
        
        insights = {
            'trends': self._calculate_trends(df),
            'top_skills': self._analyze_top_skills(comparisons),
            'success_factors': self._identify_success_factors(df),
            'recommendations': self._generate_recommendations(df)
        }
        
        return insights
    
    def _calculate_trends(self, df: pd.DataFrame) -> Dict:
        """Calculate score trends over time"""
        if len(df) < 2:
            return {
                'direction': 'stable',
                'change': 0.0,
                'weekly_averages': {}
            }
        
        df['week'] = df['date'].dt.to_period('W')
        weekly_avg = df.groupby('week')['match_score'].mean()
        
        if len(weekly_avg) < 2:
            return {
                'direction': 'stable',
                'change': 0.0,
                'weekly_averages': weekly_avg.to_dict()
            }
        
        trend = "improving" if weekly_avg.iloc[-1] > weekly_avg.iloc[0] else "declining"
        
        return {
            'direction': trend,
            'change': float(weekly_avg.iloc[-1] - weekly_avg.iloc[0]),
            'weekly_averages': weekly_avg.to_dict()
        }
    
    def _analyze_top_skills(self, comparisons: List[Dict]) -> Dict:
        """Analyze most common skills in successful matches"""
        all_skills = []
        successful_skills = []
        
        for comp in comparisons:
            if comp.get('parsed_data'):
                job_data = comp['parsed_data'].get('job', {})
                resume_data = comp['parsed_data'].get('resume', {})
                
                # Collect all skills
                job_skills = job_data.get('required_skills', [])
                resume_skills = resume_data.get('skills', [])
                
                all_skills.extend([s.lower() for s in job_skills + resume_skills])
                
                # Collect skills from successful matches (80%+)
                if comp.get('match_score', 0) >= 80:
                    successful_skills.extend([s.lower() for s in resume_skills])
        
        # Count skills
        skill_counts = Counter(all_skills)
        successful_skill_counts = Counter(successful_skills)
        
        return {
            'most_common': skill_counts.most_common(10),
            'successful_skills': successful_skill_counts.most_common(10)
        }
    
    def _identify_success_factors(self, df: pd.DataFrame) -> Dict:
        """Identify factors that contribute to high match scores"""
        high_scores = df[df['match_score'] >= 80]
        low_scores = df[df['match_score'] < 60]
        
        factors = {}
        
        if len(high_scores) > 0:
            factors['high_score_avg'] = high_scores['match_score'].mean()
            factors['high_score_count'] = len(high_scores)
        
        if len(low_scores) > 0:
            factors['low_score_avg'] = low_scores['match_score'].mean()
            factors['low_score_count'] = len(low_scores)
        
        # Calculate success rate
        total = len(df)
        if total > 0:
            factors['success_rate'] = len(high_scores) / total * 100
            factors['improvement_rate'] = len(low_scores) / total * 100
        
        return factors
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations based on data"""
        recommendations = []
        
        avg_score = df['match_score'].mean()
        
        if avg_score < 70:
            recommendations.append("Consider focusing on skill development - average scores are below target")
        
        if len(df) < 5:
            recommendations.append("Add more comparisons to get better insights")
        
        # Check for recent improvements
        recent = df.tail(5)
        if len(recent) >= 3:
            recent_avg = recent['match_score'].mean()
            if recent_avg > avg_score:
                recommendations.append("Great progress! Recent scores are improving")
            elif recent_avg < avg_score:
                recommendations.append("Recent scores are declining - review your approach")
        
        return recommendations
    
    def create_trend_chart(self, comparisons: List[Dict]) -> go.Figure:
        """Create trend visualization"""
        if not comparisons:
            # Create empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for trends",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(height=400, title="Match Score Trends")
            return fig
        
        df = pd.DataFrame(comparisons)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['match_score'],
            mode='markers+lines',
            name='Match Scores',
            line=dict(color='#667eea', width=2),
            marker=dict(size=8, color=df['match_score'], 
                       colorscale='Viridis', showscale=True)
        ))
        
        # Add trend line if enough data points
        if len(df) >= 2:
            z = np.polyfit(range(len(df)), df['match_score'], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=p(range(len(df))),
                mode='lines',
                name='Trend',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title="📈 Match Score Trends Over Time",
            xaxis_title="Date",
            yaxis_title="Match Score (%)",
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    def create_skills_analysis_chart(self, comparisons: List[Dict]) -> go.Figure:
        """Create skills analysis visualization"""
        if not comparisons:
            # Create empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for skills analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(height=400, title="Top Skills Analysis")
            return fig
        
        skills_data = self._analyze_top_skills(comparisons)
        
        if not skills_data['most_common']:
            # Create empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No skills data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(height=400, title="Top Skills Analysis")
            return fig
        
        # Prepare data for chart
        skills, counts = zip(*skills_data['most_common'][:10])
        
        fig = go.Figure(data=[
            go.Bar(
                x=skills,
                y=counts,
                marker_color='#667eea',
                text=counts,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="🎯 Most Common Skills in Comparisons",
            xaxis_title="Skills",
            yaxis_title="Frequency",
            height=400,
            xaxis_tickangle=-45
        )
        
        return fig

# Initialize analytics engine globally
analytics = AnalyticsEngine()

# Batch Processing Module
class BatchProcessor:
    def __init__(self):
        self.progress_callback = None
        
    async def batch_compare(self, job_files: List, resume_files: List, 
                           progress_callback=None) -> pd.DataFrame:
        """Perform batch comparisons with all combinations"""
        self.progress_callback = progress_callback
        total_comparisons = len(job_files) * len(resume_files)
        results = []
        
        # Process all combinations
        current = 0
        for job_file in job_files:
            try:
                job_text = extract_text_from_file(job_file)
                job_parsed, job_confidence = await cached_multi_model_parse(job_text, 'job')
                
                # Validate job data
                job_validation = validator.validate_job(job_parsed)
                if job_validation.sanitized_data:
                    job_parsed = job_validation.sanitized_data
                
                for resume_file in resume_files:
                    current += 1
                    if self.progress_callback:
                        self.progress_callback(current / total_comparisons)
                    
                    try:
                        resume_text = extract_text_from_file(resume_file)
                        resume_parsed, resume_confidence = await cached_multi_model_parse(resume_text, 'resume')
                        
                        # Validate resume data
                        resume_validation = validator.validate_resume(resume_parsed)
                        if resume_validation.sanitized_data:
                            resume_parsed = resume_validation.sanitized_data
                        
                        score, comparison = calculate_match_score(job_parsed, resume_parsed)
                        
                        results.append({
                            'job_file': job_file.name,
                            'resume_file': resume_file.name,
                            'job_title': job_parsed.get('title', 'Unknown'),
                            'candidate': resume_parsed.get('name', 'Unknown'),
                            'match_score': score,
                            'job_confidence': job_confidence,
                            'resume_confidence': resume_confidence,
                            'top_missing_skills': comparison.skill_matches_detailed['missing'][:3],
                            'matched_skills_count': len(comparison.skill_matches_detailed['matched']),
                            'missing_skills_count': len(comparison.skill_matches_detailed['missing']),
                            'overall_summary': comparison.overall_summary,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                    except Exception as e:
                        # Log error and continue with next file
                        results.append({
                            'job_file': job_file.name,
                            'resume_file': resume_file.name,
                            'job_title': 'Error',
                            'candidate': 'Error',
                            'match_score': 0,
                            'job_confidence': 0,
                            'resume_confidence': 0,
                            'top_missing_skills': [],
                            'matched_skills_count': 0,
                            'missing_skills_count': 0,
                            'overall_summary': f'Error: {str(e)}',
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                        
            except Exception as e:
                # Handle job file errors
                for resume_file in resume_files:
                    current += 1
                    if self.progress_callback:
                        self.progress_callback(current / total_comparisons)
                    
                    results.append({
                        'job_file': job_file.name,
                        'resume_file': resume_file.name,
                        'job_title': 'Error',
                        'candidate': 'Error',
                        'match_score': 0,
                        'job_confidence': 0,
                        'resume_confidence': 0,
                        'top_missing_skills': [],
                        'matched_skills_count': 0,
                        'missing_skills_count': 0,
                        'overall_summary': f'Job Error: {str(e)}',
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
        
        return pd.DataFrame(results)
    
    def create_batch_summary(self, results_df: pd.DataFrame) -> Dict:
        """Create summary statistics for batch results"""
        if results_df.empty:
            return {}
        
        summary = {
            'total_comparisons': len(results_df),
            'average_score': results_df['match_score'].mean(),
            'max_score': results_df['match_score'].max(),
            'min_score': results_df['match_score'].min(),
            'high_scores': len(results_df[results_df['match_score'] >= 80]),
            'medium_scores': len(results_df[(results_df['match_score'] >= 60) & (results_df['match_score'] < 80)]),
            'low_scores': len(results_df[results_df['match_score'] < 60]),
            'success_rate': len(results_df[results_df['match_score'] >= 80]) / len(results_df) * 100,
            'top_jobs': results_df.groupby('job_title')['match_score'].mean().nlargest(5).to_dict(),
            'top_candidates': results_df.groupby('candidate')['match_score'].mean().nlargest(5).to_dict()
        }
        
        return summary
    
    def create_batch_visualization(self, results_df: pd.DataFrame) -> go.Figure:
        """Create visualization for batch results"""
        if results_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No batch results to visualize",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(height=400, title="Batch Results")
            return fig
        
        # Create score distribution
        fig = go.Figure()
        
        # Score distribution
        fig.add_trace(go.Histogram(
            x=results_df['match_score'],
            nbinsx=20,
            name='Score Distribution',
            marker_color='#667eea',
            opacity=0.7
        ))
        
        fig.update_layout(
            title="📊 Batch Comparison Score Distribution",
            xaxis_title="Match Score (%)",
            yaxis_title="Number of Comparisons",
            height=400,
            showlegend=False
        )
        
        return fig

# Initialize batch processor globally
batch_processor = BatchProcessor()

# Gamification Engine
class GamificationEngine:
    def __init__(self):
        self.achievements = {
            'first_match': {'name': '🎯 First Match', 'desc': 'Complete your first comparison'},
            'high_scorer': {'name': '⭐ High Scorer', 'desc': 'Achieve 80%+ match'},
            'perfectionist': {'name': '💯 Perfectionist', 'desc': 'Achieve 95%+ match'},
            'skill_master': {'name': '🎓 Skill Master', 'desc': 'Match 20+ skills'},
            'consistent': {'name': '📈 Consistent', 'desc': '5 comparisons in a week'},
            'explorer': {'name': '🔍 Explorer', 'desc': 'Try 5 different job types'},
            'batch_processor': {'name': '🚀 Batch Master', 'desc': 'Complete a batch comparison'},
            'resume_optimizer': {'name': '📝 Resume Guru', 'desc': 'Upload 10+ resumes'},
            'job_collector': {'name': '💼 Job Hunter', 'desc': 'Add 10+ job descriptions'},
            'sharing_expert': {'name': '📤 Sharing Pro', 'desc': 'Share 5+ comparisons'},
            'validation_master': {'name': '✅ Quality Expert', 'desc': 'Have 10+ high-confidence parses'},
            'recommendation_follower': {'name': '🎯 Goal Setter', 'desc': 'Follow 5+ recommendations'}
        }
        
    def calculate_user_stats(self, comparisons: List[Dict]) -> Dict:
        """Calculate user statistics and achievements"""
        if not comparisons:
            return {
                'level': 1, 
                'xp': 0, 
                'achievements': [],
                'total_comparisons': 0,
                'average_score': 0,
                'best_match': None,
                'improvement_rate': 0,
                'next_level_xp': 100
            }
        
        # Calculate basic stats
        scores = [c.get('match_score', 0) for c in comparisons]
        stats = {
            'total_comparisons': len(comparisons),
            'average_score': np.mean(scores) if scores else 0,
            'best_match': max(comparisons, key=lambda x: x.get('match_score', 0)) if comparisons else None,
            'improvement_rate': self._calculate_improvement(comparisons),
            'level': self._calculate_level(comparisons),
            'xp': self._calculate_xp(comparisons),
            'next_level_xp': self._xp_for_next_level(comparisons)
        }
        
        # Check achievements
        stats['achievements'] = self._check_achievements(comparisons)
        
        return stats
    
    def _calculate_level(self, comparisons: List[Dict]) -> int:
        """Calculate user level based on activity"""
        xp = self._calculate_xp(comparisons)
        return int(np.sqrt(xp / 100)) + 1
    
    def _calculate_xp(self, comparisons: List[Dict]) -> int:
        """Calculate experience points"""
        xp = 0
        for comp in comparisons:
            xp += 10  # Base XP for each comparison
            score = comp.get('match_score', 0)
            xp += int(score / 10)  # Bonus for high scores
            
            # Bonus for high confidence scores
            if comp.get('parsed_data'):
                job_conf = comp.get('job_confidence', 0)
                resume_conf = comp.get('resume_confidence', 0)
                if job_conf and resume_conf:
                    avg_conf = (job_conf + resume_conf) / 2
                    xp += int(avg_conf * 20)  # Bonus for high confidence
            
            # Bonus for skill matches
            if comp.get('comparison_results'):
                skill_details = comp['comparison_results'].get('skill_matches_detailed', {})
                matched_skills = len(skill_details.get('matched', []))
                xp += matched_skills * 2  # 2 XP per matched skill
        
        return xp
    
    def _xp_for_next_level(self, comparisons: List[Dict]) -> int:
        """Calculate XP needed for next level"""
        current_level = self._calculate_level(comparisons)
        return (current_level ** 2) * 100
    
    def _calculate_improvement(self, comparisons: List[Dict]) -> float:
        """Calculate improvement rate over time"""
        if len(comparisons) < 2:
            return 0.0
        
        # Sort by date
        sorted_comps = sorted(comparisons, key=lambda x: x.get('date', ''))
        
        # Calculate trend
        scores = [c.get('match_score', 0) for c in sorted_comps]
        if len(scores) >= 2:
            recent_avg = np.mean(scores[-5:]) if len(scores) >= 5 else scores[-1]
            early_avg = np.mean(scores[:5]) if len(scores) >= 5 else scores[0]
            return recent_avg - early_avg
        
        return 0.0
    
    def _check_achievements(self, comparisons: List[Dict]) -> List[Dict]:
        """Check which achievements user has earned"""
        earned = []
        
        # First match
        if len(comparisons) >= 1:
            earned.append(self.achievements['first_match'])
        
        # High scorer
        if any(c.get('match_score', 0) >= 80 for c in comparisons):
            earned.append(self.achievements['high_scorer'])
        
        # Perfectionist
        if any(c.get('match_score', 0) >= 95 for c in comparisons):
            earned.append(self.achievements['perfectionist'])
        
        # Skill master
        for comp in comparisons:
            if comp.get('comparison_results'):
                skill_details = comp['comparison_results'].get('skill_matches_detailed', {})
                if len(skill_details.get('matched', [])) >= 20:
                    earned.append(self.achievements['skill_master'])
                    break
        
        # Consistent (5 comparisons in a week)
        if len(comparisons) >= 5:
            # Check if 5 comparisons were done within a week
            recent_comps = sorted(comparisons, key=lambda x: x.get('date', ''), reverse=True)[:5]
            if len(recent_comps) >= 5:
                earned.append(self.achievements['consistent'])
        
        # Explorer (5 different job types)
        job_types = set()
        for comp in comparisons:
            if comp.get('parsed_data') and comp['parsed_data'].get('job'):
                job_type = comp['parsed_data']['job'].get('type', 'Unknown')
                job_types.add(job_type)
        if len(job_types) >= 5:
            earned.append(self.achievements['explorer'])
        
        return earned
    
    def get_level_title(self, level: int) -> str:
        """Get title for user level"""
        titles = {
            1: "🚀 Beginner",
            2: "📚 Learner", 
            3: "🎯 Apprentice",
            4: "💼 Professional",
            5: "⭐ Expert",
            6: "🏆 Master",
            7: "👑 Grandmaster",
            8: "🌟 Legend",
            9: "💎 Diamond",
            10: "👽 Ultimate"
        }
        return titles.get(level, f"Level {level}")
    
    def get_next_achievements(self, comparisons: List[Dict]) -> List[Dict]:
        """Get achievements user is close to earning"""
        earned_ids = {a['name'] for a in self._check_achievements(comparisons)}
        close_achievements = []
        
        # Check for close achievements
        total_comps = len(comparisons)
        if total_comps == 0:
            close_achievements.append({
                'name': '🎯 First Match',
                'desc': 'Complete your first comparison',
                'progress': 0,
                'target': 1
            })
        
        # Check for high scorer
        max_score = max([c.get('match_score', 0) for c in comparisons]) if comparisons else 0
        if max_score < 80:
            close_achievements.append({
                'name': '⭐ High Scorer',
                'desc': 'Achieve 80%+ match',
                'progress': max_score,
                'target': 80
            })
        
        # Check for skill master
        max_skills = 0
        for comp in comparisons:
            if comp.get('comparison_results'):
                skill_details = comp['comparison_results'].get('skill_matches_detailed', {})
                max_skills = max(max_skills, len(skill_details.get('matched', [])))
        
        if max_skills < 20:
            close_achievements.append({
                'name': '🎓 Skill Master',
                'desc': 'Match 20+ skills',
                'progress': max_skills,
                'target': 20
            })
        
        return close_achievements

# Initialize gamification engine globally
gamification = GamificationEngine()

# Utility function for safe metric display
def safe_metric(label, value, delta=None, delta_color=None, **kwargs):
    """Wrapper for st.metric with parameter validation"""
    
    # Validate delta_color
    valid_delta_colors = ['normal', 'inverse', 'off', None]
    if delta_color not in valid_delta_colors:
        logger.warning(f"Invalid delta_color '{delta_color}', using default")
        delta_color = None
    
    # Create metric with validated parameters
    if delta is not None and delta_color is not None:
        return st.metric(label, value, delta=delta, delta_color=delta_color, **kwargs)
    elif delta is not None:
        return st.metric(label, value, delta=delta, **kwargs)
    else:
        return st.metric(label, value, **kwargs)

# Visualization Functions
def create_match_visualization(comparison_result: ComparisonResult) -> go.Figure:
    """Create an interactive radar chart for match visualization"""
    
    categories = ['Skills', 'Experience', 'Education', 'Keywords', 'Overall']
    scores = [
        comparison_result.ai_confidence_scores.get('skills', 0),
        comparison_result.ai_confidence_scores.get('experience', 0),
        comparison_result.ai_confidence_scores.get('education', 0),
        len(comparison_result.skill_matches_detailed['matched']) / 
        (len(comparison_result.skill_matches_detailed['matched']) + 
         len(comparison_result.skill_matches_detailed['missing'])) * 100 if (len(comparison_result.skill_matches_detailed['matched']) + 
         len(comparison_result.skill_matches_detailed['missing'])) > 0 else 0,
        comparison_result.match_score
    ]
    
    fig = go.Figure()
    
    # Add trace
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name='Match Profile',
        fillcolor='rgba(0, 102, 204, 0.2)',
        line=dict(color='rgb(0, 102, 204)', width=2),
        hovertemplate='%{theta}: %{r:.1f}%<extra></extra>'
    ))
    
    # Add threshold line
    fig.add_trace(go.Scatterpolar(
        r=[80] * len(categories),
        theta=categories,
        mode='lines',
        name='Target Score',
        line=dict(color='green', width=1, dash='dot'),
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix='%'
            )),
        showlegend=True,
        height=400,
        title="Match Analysis Radar"
    )
    
    return fig

def create_skill_sunburst(skill_details: Dict) -> go.Figure:
    """Interactive sunburst showing skill categories and gaps"""
    
    # Categorize skills (you'd expand this logic)
    skill_categories = {
        'Technical': ['python', 'java', 'sql', 'api', 'cloud', 'javascript', 'react', 'node', 'docker', 'kubernetes'],
        'Soft Skills': ['communication', 'leadership', 'teamwork', 'problem solving', 'analytical', 'collaboration'],
        'Tools': ['git', 'docker', 'kubernetes', 'aws', 'azure', 'jenkins', 'jira', 'confluence'],
        'Languages': ['english', 'spanish', 'french', 'german', 'mandarin', 'japanese'],
        'Certifications': ['pmp', 'scrum', 'aws', 'azure', 'google cloud', 'cisco']
    }
    
    data = []
    for category, keywords in skill_categories.items():
        matched = [s for s in skill_details['matched'] if any(k in s.lower() for k in keywords)]
        missing = [s for s in skill_details['missing'] if any(k in s.lower() for k in keywords)]
        
        data.append(dict(labels=category, parents="", values=len(matched) + len(missing)))
        if matched:
            data.append(dict(labels="✓ Matched", parents=category, values=len(matched)))
        if missing:
            data.append(dict(labels="✗ Missing", parents=category, values=len(missing)))
    
    # Add uncategorized skills
    all_skills = set(skill_details['matched'] + skill_details['missing'])
    categorized_skills = set()
    for category, keywords in skill_categories.items():
        for skill in all_skills:
            if any(k in skill.lower() for k in keywords):
                categorized_skills.add(skill)
    
    uncategorized = all_skills - categorized_skills
    if uncategorized:
        data.append(dict(labels="Other", parents="", values=len(uncategorized)))
        matched_other = [s for s in skill_details['matched'] if s in uncategorized]
        missing_other = [s for s in skill_details['missing'] if s in uncategorized]
        if matched_other:
            data.append(dict(labels="✓ Matched", parents="Other", values=len(matched_other)))
        if missing_other:
            data.append(dict(labels="✗ Missing", parents="Other", values=len(missing_other)))
    
    if not data:
        # Fallback if no skills found
        data = [
            dict(labels="No Skills", parents="", values=1),
            dict(labels="No Data", parents="No Skills", values=1)
        ]
    
    fig = go.Figure(go.Sunburst(
        labels=[d['labels'] for d in data],
        parents=[d['parents'] for d in data],
        values=[d['values'] for d in data],
        branchvalues="total",
        marker=dict(colors=['green' if '✓' in l else 'red' if '✗' in l else 'blue' 
                           for l in [d['labels'] for d in data]])
    ))
    
    fig.update_layout(height=400, title="Skills Distribution")
    return fig

def create_skill_gap_chart(skill_details: Dict) -> go.Figure:
    """Create a bar chart showing skill gaps"""
    
    matched_count = len(skill_details['matched'])
    missing_count = len(skill_details['missing'])
    total_count = matched_count + missing_count
    
    if total_count == 0:
        # Create empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="No skills data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=300, title="Skills Gap Analysis")
        return fig
    
    categories = ['Matched Skills', 'Missing Skills']
    values = [matched_count, missing_count]
    colors = ['green', 'red']
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Skills Gap Analysis",
        xaxis_title="Skill Status",
        yaxis_title="Number of Skills",
        height=300,
        showlegend=False
    )
    
    return fig

# Initialize session state
def init_session_state():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Dashboard'
    if 'wizard_stage' not in st.session_state:
        st.session_state.wizard_stage = 1
    if 'parsed_job' not in st.session_state:
        st.session_state.parsed_job = None
    if 'parsed_resume' not in st.session_state:
        st.session_state.parsed_resume = None
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = None
    if 'job_confidence' not in st.session_state:
        st.session_state.job_confidence = None
    if 'resume_confidence' not in st.session_state:
        st.session_state.resume_confidence = None
    if 'show_comparison_detail' not in st.session_state:
        st.session_state.show_comparison_detail = False
    if 'selected_comparison' not in st.session_state:
        st.session_state.selected_comparison = None
    if 'settings' not in st.session_state:
        st.session_state.settings = load_settings()
    # Initialize AI processing progress states
    if 'job_progress' not in st.session_state:
        st.session_state.job_progress = StreamingProgress()
    if 'resume_progress' not in st.session_state:
        st.session_state.resume_progress = StreamingProgress()
    
    # Initialize new AI components
    if 'prompt_engineer' not in st.session_state:
        st.session_state.prompt_engineer = AdvancedPromptEngineer()
    if 'question_generator' not in st.session_state:
        st.session_state.question_generator = DynamicQuestionGenerator()
    if 'tip_generator' not in st.session_state:
        st.session_state.tip_generator = ContextualTipGenerator()

# Settings management
def load_settings():
    default_settings = {
        "theme": "light",
        "default_view": "grid",
        "skill_weight": 40,
        "experience_weight": 30,
        "education_weight": 15,
        "other_weight": 15,
        "llm_blend": 30,
        "model": "llama3",
        "temperature": 0.1,
        "export_format": "pdf"
    }
    
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return {**default_settings, **json.load(f)}
        except:
            return default_settings
    return default_settings

def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)

# File processing
def extract_text_from_pdf(file) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        raise Exception(f"Failed to extract text from PDF: {str(e)}")

def extract_text_from_docx(file) -> str:
    try:
        doc = Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    except Exception as e:
        logger.error(f"DOCX extraction error: {e}")
        raise Exception(f"Failed to extract text from DOCX: {str(e)}")

def extract_text_from_file(file) -> str:
    filename = file.name.lower()
    
    if filename.endswith('.pdf'):
        return extract_text_from_pdf(file)
    elif filename.endswith('.docx'):
        return extract_text_from_docx(file)
    elif filename.endswith('.txt'):
        return file.read().decode('utf-8')
    else:
        raise ValueError(f"Unsupported file type: {filename}")

# LLM Integration
def check_ollama_connection():
    """Check if SmolLM2 model is available"""
    try:
        # Suppress transformers warnings during import
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Try to load tokenizer as a quick check
        # Use a simpler check that doesn't fully load the model
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "HuggingFaceTB/SmolLM2-135M", 
                trust_remote_code=True,
                local_files_only=False  # Allow downloading if needed
            )
            return True, ["SmolLM2-135M"]
        except Exception as e:
            logger.warning(f"SmolLM2 model check failed: {e}")
            # Return True anyway to allow the app to run
            return True, ["SmolLM2-135M"]
            
    except Exception as e:
        logger.error(f"SmolLM2 not available: {e}")
        return False, []

class AdvancedLLMParser:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.prompt_templates = {
            'structured': {
                'job': """Extract the following information from this job description and return ONLY a JSON object. Be thorough in extracting ALL information:
{{
    "title": "exact job title",
    "company": "company name",
    "location": "city, state or remote",
    "type": "Full-time or Part-time or Contract or Remote",
    "responsibilities": ["list ALL job responsibilities", "each as separate item"],
    "required_skills": ["list ALL required skills", "include technical skills", "programming languages", "tools", "frameworks"],
    "preferred_skills": ["list preferred/nice-to-have skills"],
    "education": "education requirements (e.g., Bachelor's degree in Computer Science)",
    "experience": "experience requirements (e.g., 5+ years experience)"
}}

IMPORTANT: 
- Extract ALL skills mentioned in requirements, qualifications, or skills sections
- Include both technical and soft skills
- For experience, look for phrases like "X years", "experience required", etc.
- For education, look for degree requirements, certifications, etc.

Job Description:
{text}

Return ONLY the JSON object, no explanations.""",
                'resume': """Extract the following information from this resume and return ONLY a JSON object:
{{
    "name": "full name",
    "email": "email address",
    "phone": "phone number (include country code if present)",
    "summary": "professional summary or objective",
    "experience": [
        {{
            "title": "exact job title",
            "company": "company name",
            "duration": "dates or duration (e.g., Jan 2020 - Dec 2022)",
            "responsibilities": ["list key responsibilities", "achievements", "technologies used"]
        }}
    ],
    "skills": ["list ALL technical skills", "programming languages", "frameworks", "tools", "soft skills"],
    "education": "highest degree and institution (e.g., Bachelor of Science in Computer Science from MIT)"
}}

IMPORTANT:
- Extract ALL skills from skills section, experience descriptions, and projects
- Include complete work history with dates
- Capture education details including degree, major, and institution

Resume:
{text}

Return ONLY the JSON object, no explanations."""
            },
            'conversational': {
                'job': """Please help me extract all the important information from this job description. I need:
- Job title
- Company name  
- Location (city/state or if remote)
- Employment type (full-time, part-time, contract, etc.)
- ALL required skills and qualifications
- Preferred skills if mentioned
- Education requirements
- Years of experience required
- Main responsibilities

Make sure to capture ALL skills mentioned anywhere in the description.

Job Description:
{text}

Please return the information as a JSON object with these fields: title, company, location, type, responsibilities, required_skills, preferred_skills, education, experience.""",
                'resume': """Please help me extract all information from this resume including:
- Full name
- Email and phone
- Professional summary
- Complete work experience with titles, companies, dates, and what they did
- ALL technical and soft skills
- Education details including degree and school
- Any other relevant information

Resume:
{text}

Please return as JSON with fields: name, email, phone, summary, experience, skills, education."""
            },
            'analytical': {
                'job': """Analyze this job description comprehensively. Extract and categorize:
1. Basic info: title, company, location, employment type
2. Requirements: all required skills, minimum experience, education
3. Preferences: nice-to-have skills, preferred qualifications
4. Responsibilities: main duties and expectations

Pay special attention to:
- Technical requirements (languages, frameworks, tools)
- Experience level (years required)
- Educational requirements
- Soft skills mentioned

Job Description:
{text}

Return structured JSON with all extracted information.""",
                'resume': """Analyze this resume comprehensively. Extract:
1. Contact info: name, email, phone
2. Professional profile/summary
3. Complete employment history with dates and achievements
4. Full skills inventory (technical and soft skills)
5. Educational background
6. Any additional relevant information

Resume:
{text}

Return complete JSON with all information."""
            }
        }
    
    async def multi_model_parse(self, text: str, parse_type: str, models: List[str] = None) -> Tuple[Dict, float]:
        """Parse with multiple models and return consensus with confidence score"""
        if not models:
            # Check available models
            connected, available_models = check_ollama_connection()
            if connected and available_models:
                models = available_models[:2]  # Use top 2 available
            else:
                # No models available, return empty result and low confidence
                return {"error": "No AI model available"}, 0.0
        
        tasks = []
        for model in models:
            for prompt_style in self.prompt_templates:
                task = asyncio.create_task(self._parse_with_model(text, parse_type, model, prompt_style))
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        valid_results = [r for r in results if isinstance(r, dict)]
        
        if not valid_results:
            return {"error": "No valid AI parse results"}, 0.0
        
        # Calculate consensus
        consensus_result = self._calculate_consensus(valid_results)
        confidence = self._calculate_confidence(valid_results)
        
        return consensus_result, confidence
    
    async def _parse_with_model(self, text: str, parse_type: str, model: str, prompt_style: str) -> Dict:
        """Parse text with SmolLM2 model"""
        try:
            # Use the llm_parse function we updated
            result = llm_parse(text, parse_type, model)
            return result
        except Exception as e:
            logger.error(f"SmolLM2 parsing error: {e}")
            return {}
    
    def _calculate_consensus(self, results: List[Dict]) -> Dict:
        """Merge results using voting mechanism"""
        consensus = {}
        
        # For each field, take the most common non-empty value
        all_keys = set()
        for r in results:
            all_keys.update(r.keys())
        
        for key in all_keys:
            values = [r.get(key) for r in results if r.get(key)]
            if values:
                if isinstance(values[0], list):
                    # For lists, merge and deduplicate
                    consensus[key] = list(set(sum(values, [])))
                else:
                    # For strings, take most common
                    consensus[key] = Counter(values).most_common(1)[0][0]
        
        return consensus
    
    def _calculate_confidence(self, results: List[Dict]) -> float:
        """Calculate confidence based on agreement between models"""
        if len(results) < 2:
            return 0.5
        
        # Calculate similarity between results
        similarities = []
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                sim = self._dict_similarity(results[i], results[j])
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _dict_similarity(self, dict1: Dict, dict2: Dict) -> float:
        """Calculate similarity between two dictionaries"""
        if not dict1 or not dict2:
            return 0.0
        
        all_keys = set(dict1.keys()) | set(dict2.keys())
        if not all_keys:
            return 1.0
        
        matches = 0
        for key in all_keys:
            val1 = dict1.get(key)
            val2 = dict2.get(key)
            
            if val1 == val2:
                matches += 1
            elif isinstance(val1, list) and isinstance(val2, list):
                # For lists, check if they have common elements
                if set(val1) & set(val2):
                    matches += 0.5
        
        return matches / len(all_keys)

# ====================================================================================
# SECTION 1: ADVANCED PROMPT ENGINEERING MODULE
# ====================================================================================

class AdvancedPromptEngineer:
    """Advanced prompt engineering with Chain-of-Thought, few-shot examples, and self-consistency"""
    
    def __init__(self):
        self.few_shot_examples = {
            'job_parsing': [
                {
                    'input': "Senior Python Developer at TechCorp. 5+ years experience required. Skills: Python, Django, AWS.",
                    'reasoning': "First, I identify the job title 'Senior Python Developer'. Company is 'TechCorp'. Experience requirement is clearly stated as '5+ years'. Required skills are listed after 'Skills:' - Python, Django, AWS.",
                    'output': {
                        "title": "Senior Python Developer",
                        "company": "TechCorp",
                        "experience": "5+ years",
                        "required_skills": ["Python", "Django", "AWS"]
                    }
                }
            ],
            'resume_parsing': [
                {
                    'input': "John Doe | john@email.com | Python Developer with 5 years experience in Django and React",
                    'reasoning': "Name is at the start 'John Doe'. Email follows the name 'john@email.com'. The summary mentions 'Python Developer' with '5 years experience'. Skills mentioned are Django and React.",
                    'output': {
                        "name": "John Doe",
                        "email": "john@email.com",
                        "summary": "Python Developer with 5 years experience in Django and React",
                        "skills": ["Python", "Django", "React"]
                    }
                }
            ]
        }
        
        self.cot_templates = {
            'analyze': """Let me analyze this step by step:
1. First, I'll identify the document type and structure
2. Then, I'll extract the key sections
3. Next, I'll parse each field carefully
4. Finally, I'll validate my findings

Thinking process:
{analysis}

Based on this analysis:
{conclusion}""",
            
            'extract': """To extract {field} from this text, I need to:
1. Look for keywords like {keywords}
2. Check common patterns
3. Consider context clues
4. Validate the extracted value

Let me search:
{search_process}

Found: {result}""",
            
            'validate': """I'll validate this extraction by:
1. Checking if it makes logical sense
2. Verifying format correctness
3. Comparing with similar examples
4. Ensuring consistency

Validation:
{validation_steps}

Confidence: {confidence}%"""
        }
    
    def create_chain_of_thought_prompt(self, text: str, task: str, examples: bool = True) -> str:
        """Create a Chain-of-Thought prompt with reasoning steps"""
        
        base_prompt = f"Task: {task}\n\n"
        
        # Add few-shot examples if requested
        if examples and task in self.few_shot_examples:
            base_prompt += "Here are some examples of how to approach this:\n\n"
            for i, example in enumerate(self.few_shot_examples[task][:2], 1):
                base_prompt += f"Example {i}:\n"
                base_prompt += f"Input: {example['input']}\n"
                base_prompt += f"Reasoning: {example['reasoning']}\n"
                base_prompt += f"Output: {json.dumps(example['output'], indent=2)}\n\n"
        
        # Add Chain-of-Thought instruction
        base_prompt += """Now, let's analyze the given text step by step:

Step 1: Identify the document structure
Step 2: Locate key information sections  
Step 3: Extract relevant fields
Step 4: Validate and format the data

Text to analyze:
{text}

Let me think through this carefully:
"""
        
        return base_prompt.format(text=text[:1000])  # Limit text length
    
    def create_self_consistency_prompts(self, text: str, task: str, variations: int = 3) -> List[str]:
        """Create multiple prompt variations for self-consistency checking"""
        
        prompts = []
        
        # Variation 1: Direct extraction
        prompts.append(f"""Extract {task} from this text. Be precise and thorough.

Text: {text}

Extracted data:""")
        
        # Variation 2: Step-by-step analysis
        prompts.append(f"""Analyze this text step-by-step to extract {task}.

Step 1: Read the entire text
Step 2: Identify relevant sections
Step 3: Extract required information
Step 4: Format the results

Text: {text}

Analysis and extraction:""")
        
        # Variation 3: Question-based approach
        prompts.append(f"""Answer these questions about the text to extract {task}:
- What is the main subject?
- What key information is provided?
- What specific details are mentioned?

Text: {text}

Answers and extracted data:""")
        
        return prompts[:variations]
    
    def create_role_specific_prompt(self, job_data: Dict, resume_data: Dict, task: str) -> str:
        """Create prompts specific to the job role and candidate profile"""
        
        job_title = job_data.get('title', 'Unknown Position')
        skills_required = job_data.get('required_skills', [])
        candidate_skills = resume_data.get('skills', [])
        
        if task == "interview_questions":
            return f"""As an experienced interviewer for a {job_title} position, create 5 role-specific interview questions.

Job requires: {', '.join(skills_required[:5])}
Candidate has: {', '.join(candidate_skills[:5])}

Focus on:
1. Technical skills gaps: {', '.join(set(skills_required) - set(candidate_skills))[:3]}
2. Relevant experience validation
3. Problem-solving scenarios for {job_title}
4. Cultural fit and soft skills
5. Growth potential

Generate thoughtful, specific questions:"""
        
        elif task == "improvement_tips":
            return f"""As a career coach, provide specific tips for improving this candidate's match for {job_title}.

Current match analysis:
- Has skills: {', '.join(set(candidate_skills) & set(skills_required))[:5]}
- Missing skills: {', '.join(set(skills_required) - set(candidate_skills))[:5]}
- Experience: {resume_data.get('experience', 'Unknown')}

Provide actionable, specific advice:"""
        
        return ""

# ====================================================================================
# SECTION 2: DYNAMIC QUESTION GENERATOR
# ====================================================================================

class DynamicQuestionGenerator:
    """AI-powered dynamic question generation based on job and resume analysis"""
    
    def __init__(self):
        self.question_categories = {
            'technical': "Technical Skills Assessment",
            'behavioral': "Behavioral Competencies",
            'situational': "Problem-Solving Scenarios",
            'experience': "Experience Validation",
            'cultural': "Cultural Fit & Values",
            'growth': "Growth Potential & Learning"
        }
        
        self.difficulty_levels = {
            'screening': "Basic screening questions",
            'intermediate': "Mid-level assessment",
            'advanced': "Senior-level evaluation",
            'expert': "Expert-level deep dive"
        }
    
    def generate_interview_questions(self, job_data: Dict, resume_data: Dict, 
                                   comparison_result: ComparisonResult,
                                   num_questions: int = 10) -> Dict[str, List[Dict]]:
        """Generate role-specific interview questions using AI"""
        
        # Initialize model if needed
        if not hasattr(self, 'model') or not hasattr(self, 'tokenizer'):
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                "HuggingFaceTB/SmolLM2-135M",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            self.model.eval()
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        questions_by_category = {}
        prompt_engineer = AdvancedPromptEngineer()
        
        # Determine difficulty level based on job title and experience
        difficulty = self._determine_difficulty(job_data, resume_data)
        
        # Generate questions for each category
        for category, description in self.question_categories.items():
            category_prompt = self._create_category_prompt(
                category, job_data, resume_data, comparison_result, difficulty
            )
            
            # Use Chain-of-Thought prompting
            cot_prompt = prompt_engineer.create_chain_of_thought_prompt(
                category_prompt, 
                f"interview_questions_{category}"
            )
            
            # Generate questions
            questions = self._generate_questions_for_category(
                cot_prompt, category, num_questions // len(self.question_categories)
            )
            
            questions_by_category[category] = questions
        
        # Add follow-up questions for key areas
        follow_ups = self._generate_follow_up_questions(
            job_data, resume_data, comparison_result, questions_by_category
        )
        
        questions_by_category['follow_ups'] = follow_ups
        
        return questions_by_category
    
    def _determine_difficulty(self, job_data: Dict, resume_data: Dict) -> str:
        """Determine appropriate difficulty level"""
        
        job_title = job_data.get('title', '').lower()
        experience = job_data.get('experience', '').lower()
        
        # Check for seniority indicators
        if any(term in job_title for term in ['senior', 'lead', 'principal', 'staff']):
            return 'advanced'
        elif any(term in job_title for term in ['mid', 'intermediate']):
            return 'intermediate'
        elif any(term in job_title for term in ['junior', 'entry']):
            return 'screening'
        
        # Check experience requirements
        if '7+' in experience or '10+' in experience:
            return 'expert'
        elif '5+' in experience:
            return 'advanced'
        elif '3+' in experience:
            return 'intermediate'
        
        return 'screening'
    
    def _create_category_prompt(self, category: str, job_data: Dict, 
                               resume_data: Dict, comparison_result: ComparisonResult,
                               difficulty: str) -> str:
        """Create category-specific prompt"""
        
        job_title = job_data.get('title', 'Unknown Position')
        missing_skills = comparison_result.skill_matches_detailed.get('missing', [])
        matched_skills = comparison_result.skill_matches_detailed.get('matched', [])
        
        base_context = f"""
Job Title: {job_title}
Difficulty Level: {difficulty}
Category: {self.question_categories[category]}
Missing Skills: {', '.join(missing_skills[:5])}
Matched Skills: {', '.join(matched_skills[:5])}
"""
        
        category_prompts = {
            'technical': f"""{base_context}
Generate technical interview questions that:
1. Assess proficiency in {', '.join(matched_skills[:3])}
2. Explore potential in {', '.join(missing_skills[:3])}
3. Test problem-solving with real scenarios
4. Evaluate technical depth for {job_title}
5. Check understanding of best practices""",
            
            'behavioral': f"""{base_context}
Generate behavioral questions using STAR method that assess:
1. Leadership and collaboration
2. Conflict resolution
3. Adaptability and learning
4. Work ethic and reliability
5. Communication skills for {job_title}""",
            
            'situational': f"""{base_context}
Create scenario-based questions for {job_title} that test:
1. Real-world problem solving
2. Decision making under pressure
3. Prioritization skills
4. Technical troubleshooting
5. Project management abilities""",
            
            'experience': f"""{base_context}
Formulate questions to validate:
1. Claimed experience in {', '.join(matched_skills[:3])}
2. Depth of knowledge
3. Project complexity handled
4. Team collaboration experience
5. Industry-specific knowledge""",
            
            'cultural': f"""{base_context}
Design questions to assess:
1. Alignment with company values
2. Team fit and collaboration style
3. Work environment preferences
4. Career motivations
5. Long-term goals""",
            
            'growth': f"""{base_context}
Create questions that explore:
1. Learning agility
2. Career aspirations
3. Skill development plans
4. Adaptability to new technologies
5. Leadership potential"""
        }
        
        return category_prompts.get(category, base_context)
    
    def _generate_questions_for_category(self, prompt: str, category: str, 
                                       num_questions: int) -> List[Dict]:
        """Generate questions using AI for a specific category"""
        
        # Add specific instruction for question generation
        full_prompt = f"""{prompt}

Generate exactly {num_questions} interview questions.
Format each question with:
- The question itself
- Why it's important to ask
- What to look for in the answer
- Potential red flags

Questions:
1."""
        
        # Generate response
        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(full_prompt):].strip()
        
        # Parse response into structured questions
        questions = self._parse_question_response(response, category, num_questions)
        
        return questions
    
    def _parse_question_response(self, response: str, category: str, 
                                expected_count: int) -> List[Dict]:
        """Parse AI response into structured question format"""
        
        questions = []
        
        # Split by numbers (1., 2., etc.)
        parts = re.split(r'\d+\.', response)
        
        for i, part in enumerate(parts[1:expected_count+1], 1):
            if not part.strip():
                continue
                
            lines = part.strip().split('\n')
            question_text = lines[0].strip() if lines else f"Question {i}"
            
            # Extract additional details if present
            importance = ""
            look_for = ""
            red_flags = ""
            
            for line in lines[1:]:
                line_lower = line.lower()
                if 'important' in line_lower or 'why' in line_lower:
                    importance = line.strip()
                elif 'look for' in line_lower or 'expect' in line_lower:
                    look_for = line.strip()
                elif 'red flag' in line_lower or 'concern' in line_lower:
                    red_flags = line.strip()
            
            questions.append({
                'question': question_text,
                'category': category,
                'importance': importance or f"Assesses {self.question_categories[category]}",
                'look_for': look_for or "Clear, specific examples and thoughtful responses",
                'red_flags': red_flags or "Vague answers, lack of examples, or contradictions",
                'difficulty': self._assess_question_difficulty(question_text),
                'time_estimate': "2-3 minutes"
            })
        
        # If we didn't get enough questions, add some fallbacks
        while len(questions) < expected_count:
            questions.append(self._generate_fallback_question(category, len(questions) + 1))
        
        return questions[:expected_count]
    
    def _assess_question_difficulty(self, question: str) -> str:
        """Assess the difficulty of a question"""
        
        # Simple heuristic based on question complexity
        if any(term in question.lower() for term in ['design', 'architect', 'optimize', 'scale']):
            return 'advanced'
        elif any(term in question.lower() for term in ['explain', 'describe', 'implement']):
            return 'intermediate'
        else:
            return 'basic'
    
    def _generate_fallback_question(self, category: str, index: int) -> Dict:
        """Generate fallback questions if AI generation fails"""
        
        fallback_questions = {
            'technical': [
                "Describe a challenging technical problem you solved recently.",
                "How do you stay updated with new technologies?",
                "Walk me through your approach to debugging complex issues."
            ],
            'behavioral': [
                "Tell me about a time you had to work with a difficult team member.",
                "Describe a situation where you had to meet a tight deadline.",
                "How do you handle constructive criticism?"
            ],
            'situational': [
                "How would you handle a production outage?",
                "What would you do if you disagreed with your manager's technical decision?",
                "How would you prioritize multiple urgent tasks?"
            ]
        }
        
        questions = fallback_questions.get(category, ["Tell me about your experience."])
        question = questions[index % len(questions)]
        
        return {
            'question': question,
            'category': category,
            'importance': f"Standard {category} assessment",
            'look_for': "Structured thinking and relevant examples",
            'red_flags': "Lack of specific examples or unclear communication",
            'difficulty': 'intermediate',
            'time_estimate': "2-3 minutes"
        }
    
    def _generate_follow_up_questions(self, job_data: Dict, resume_data: Dict,
                                    comparison_result: ComparisonResult,
                                    existing_questions: Dict) -> List[Dict]:
        """Generate smart follow-up questions based on initial responses"""
        
        follow_ups = []
        
        # Identify key areas for follow-up
        missing_skills = comparison_result.skill_matches_detailed.get('missing', [])
        
        if missing_skills:
            # Generate follow-up for skill gaps
            skill_prompt = f"""For a candidate missing {', '.join(missing_skills[:3])}, 
generate 3 follow-up questions to assess:
1. Learning potential for these skills
2. Related experience that could transfer
3. Motivation to acquire these skills

Questions:"""
            
            skill_followups = self._generate_questions_for_category(
                skill_prompt, 'follow_ups', 3
            )
            follow_ups.extend(skill_followups)
        
        return follow_ups

# ====================================================================================
# SECTION 3: CONTEXTUAL TIP GENERATOR
# ====================================================================================

class ContextualTipGenerator:
    """Generate personalized, context-aware tips based on actual content analysis"""
    
    def __init__(self):
        self.tip_categories = {
            'resume_improvement': "Resume Enhancement Tips",
            'skill_development': "Skill Development Roadmap",
            'interview_preparation': "Interview Preparation Guide",
            'networking': "Strategic Networking Advice",
            'application_strategy': "Application Strategy Tips",
            'personal_branding': "Personal Branding Recommendations"
        }
        
        # Initialize model reference
        self.model = None
        self.tokenizer = None
    
    def generate_contextual_tips(self, job_data: Dict, resume_data: Dict,
                               comparison_result: ComparisonResult) -> Dict[str, List[Dict]]:
        """Generate comprehensive contextual tips"""
        
        # Initialize model if needed
        if not self.model or not self.tokenizer:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                "HuggingFaceTB/SmolLM2-135M",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            self.model.eval()
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        tips_by_category = {}
        prompt_engineer = AdvancedPromptEngineer()
        
        # Analyze context for personalization
        context_analysis = self._analyze_context(job_data, resume_data, comparison_result)
        
        # Generate tips for each category
        for category, description in self.tip_categories.items():
            # Create role-specific prompt
            category_prompt = prompt_engineer.create_role_specific_prompt(
                job_data, resume_data, f"tips_{category}"
            )
            
            # Generate tips with context
            tips = self._generate_category_tips(
                category, context_analysis, job_data, resume_data, comparison_result
            )
            
            tips_by_category[category] = tips
        
        # Add priority action items
        priority_actions = self._generate_priority_actions(
            context_analysis, comparison_result
        )
        tips_by_category['priority_actions'] = priority_actions
        
        return tips_by_category
    
    def _analyze_context(self, job_data: Dict, resume_data: Dict, 
                        comparison_result: ComparisonResult) -> Dict:
        """Analyze context for personalized tip generation"""
        
        # Extract key insights
        job_title = job_data.get('title', 'Unknown Position')
        company = job_data.get('company', 'Unknown Company')
        match_score = comparison_result.match_score
        missing_skills = comparison_result.skill_matches_detailed.get('missing', [])
        matched_skills = comparison_result.skill_matches_detailed.get('matched', [])
        
        # Analyze career level
        career_level = 'entry'
        if any(term in job_title.lower() for term in ['senior', 'lead', 'principal']):
            career_level = 'senior'
        elif any(term in job_title.lower() for term in ['mid', 'intermediate']):
            career_level = 'mid'
        
        # Analyze skill gaps
        skill_gap_severity = 'low'
        if len(missing_skills) > len(matched_skills):
            skill_gap_severity = 'high'
        elif len(missing_skills) > 3:
            skill_gap_severity = 'medium'
        
        # Extract experience level
        experience_years = 0
        experience_text = resume_data.get('experience', [])
        if isinstance(experience_text, list) and experience_text:
            # Try to extract years from experience
            for exp in experience_text:
                duration = exp.get('duration', '') if isinstance(exp, dict) else ''
                years_match = re.search(r'(\d+)\s*year', str(duration), re.I)
                if years_match:
                    experience_years = max(experience_years, int(years_match.group(1)))
        
        return {
            'job_title': job_title,
            'company': company,
            'career_level': career_level,
            'match_score': match_score,
            'skill_gap_severity': skill_gap_severity,
            'missing_skills': missing_skills,
            'matched_skills': matched_skills,
            'experience_years': experience_years,
            'improvement_potential': 100 - match_score
        }
    
    def _generate_category_tips(self, category: str, context: Dict,
                              job_data: Dict, resume_data: Dict,
                              comparison_result: ComparisonResult) -> List[Dict]:
        """Generate tips for a specific category"""
        
        # Build category-specific prompt
        prompt = self._build_category_prompt(category, context, job_data, resume_data)
        
        # Generate tips using AI
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        # Parse response into structured tips
        tips = self._parse_tips_response(response, category, context)
        
        return tips
    
    def _build_category_prompt(self, category: str, context: Dict,
                             job_data: Dict, resume_data: Dict) -> str:
        """Build category-specific prompt for tip generation"""
        
        base_context = f"""
Position: {context['job_title']} at {context['company']}
Match Score: {context['match_score']:.1f}%
Career Level: {context['career_level']}
Experience: {context['experience_years']} years
Missing Skills: {', '.join(context['missing_skills'][:5])}
"""
        
        prompts = {
            'resume_improvement': f"""{base_context}
Generate 5 specific tips to improve this resume for {context['job_title']}:
1. How to better highlight {', '.join(context['matched_skills'][:3])}
2. How to address the gap in {', '.join(context['missing_skills'][:3])}
3. Specific keywords to add
4. Formatting improvements for ATS
5. Quantifiable achievements to emphasize

Tips:""",
            
            'skill_development': f"""{base_context}
Create a realistic skill development plan for {context['missing_skills'][:3]}:
1. Which skill to prioritize first and why
2. Free and paid learning resources
3. Practical projects to build portfolio
4. Timeline for each skill
5. How to demonstrate progress

Plan:""",
            
            'interview_preparation': f"""{base_context}
Provide interview preparation tips specific to {context['job_title']}:
1. Key topics to review
2. Projects to showcase
3. Questions to ask the interviewer
4. How to address skill gaps
5. Company-specific preparation

Tips:""",
            
            'networking': f"""{base_context}
Suggest networking strategies for {context['job_title']} role:
1. Relevant online communities
2. Key people to connect with
3. Content to share/create
4. Events to attend
5. LinkedIn optimization

Strategies:""",
            
            'application_strategy': f"""{base_context}
Optimize application strategy for {context['match_score']:.1f}% match:
1. Whether to apply now or after skill development
2. How to address gaps in cover letter
3. Portfolio pieces to include
4. Follow-up strategy
5. Alternative similar roles to consider

Strategy:""",
            
            'personal_branding': f"""{base_context}
Build personal brand for {context['job_title']} aspirations:
1. LinkedIn headline and summary
2. Portfolio website focus
3. Content creation ideas
4. Skills to highlight publicly
5. Professional story narrative

Recommendations:"""
        }
        
        return prompts.get(category, base_context)
    
    def _parse_tips_response(self, response: str, category: str, context: Dict) -> List[Dict]:
        """Parse AI response into structured tips"""
        
        tips = []
        
        # Split response into individual tips
        lines = response.split('\n')
        current_tip = []
        
        for line in lines:
            if re.match(r'^\d+\.', line.strip()):
                if current_tip:
                    # Process previous tip
                    tip_text = '\n'.join(current_tip).strip()
                    if tip_text:
                        tips.append(self._structure_tip(tip_text, category, context))
                current_tip = [line]
            else:
                current_tip.append(line)
        
        # Process last tip
        if current_tip:
            tip_text = '\n'.join(current_tip).strip()
            if tip_text:
                tips.append(self._structure_tip(tip_text, category, context))
        
        # Ensure we have at least 3 tips
        while len(tips) < 3:
            tips.append(self._generate_fallback_tip(category, context, len(tips)))
        
        return tips
    
    def _structure_tip(self, tip_text: str, category: str, context: Dict) -> Dict:
        """Structure a tip with metadata"""
        
        # Remove number prefix if present
        tip_text = re.sub(r'^\d+\.\s*', '', tip_text)
        
        # Determine priority based on context
        priority = 'medium'
        if context['match_score'] < 60:
            priority = 'high'
        elif context['match_score'] > 80:
            priority = 'low'
        
        # Estimate impact
        impact = self._estimate_tip_impact(tip_text, context)
        
        # Determine effort required
        effort = self._estimate_effort(tip_text)
        
        return {
            'category': category,
            'tip': tip_text,
            'priority': priority,
            'impact': impact,
            'effort': effort,
            'timeframe': self._estimate_timeframe(effort),
            'specific_to': context['job_title'],
            'resources': self._extract_resources(tip_text)
        }
    
    def _estimate_tip_impact(self, tip_text: str, context: Dict) -> str:
        """Estimate the impact of implementing this tip"""
        
        high_impact_keywords = ['critical', 'essential', 'must', 'significantly', 'greatly']
        medium_impact_keywords = ['important', 'helpful', 'improve', 'enhance', 'better']
        
        tip_lower = tip_text.lower()
        
        if any(word in tip_lower for word in high_impact_keywords):
            return 'high'
        elif any(word in tip_lower for word in medium_impact_keywords):
            return 'medium'
        
        # Context-based impact
        if context['skill_gap_severity'] == 'high' and 'skill' in tip_lower:
            return 'high'
        
        return 'medium'
    
    def _estimate_effort(self, tip_text: str) -> str:
        """Estimate effort required to implement tip"""
        
        low_effort_keywords = ['simple', 'quick', 'easy', 'minor', 'small']
        high_effort_keywords = ['comprehensive', 'extensive', 'complete', 'major', 'significant']
        
        tip_lower = tip_text.lower()
        
        if any(word in tip_lower for word in low_effort_keywords):
            return 'low'
        elif any(word in tip_lower for word in high_effort_keywords):
            return 'high'
        
        return 'medium'
    
    def _estimate_timeframe(self, effort: str) -> str:
        """Estimate timeframe based on effort"""
        
        timeframes = {
            'low': '1-2 days',
            'medium': '1-2 weeks',
            'high': '1-2 months'
        }
        
        return timeframes.get(effort, '1 week')
    
    def _extract_resources(self, tip_text: str) -> List[str]:
        """Extract mentioned resources from tip text"""
        
        resources = []
        
        # Look for common resource patterns
        resource_patterns = [
            r'(?:use|try|check out|visit)\s+([A-Za-z0-9\s]+)',
            r'(?:on|at|via)\s+([A-Za-z0-9\s]+(?:\.com|\.org|\.io)?)',
            r'(?:course|tutorial|guide|book|platform):\s*([^,\.]+)'
        ]
        
        for pattern in resource_patterns:
            matches = re.findall(pattern, tip_text, re.I)
            resources.extend([m.strip() for m in matches if len(m.strip()) > 2])
        
        return list(set(resources))[:3]  # Return up to 3 unique resources
    
    def _generate_fallback_tip(self, category: str, context: Dict, index: int) -> Dict:
        """Generate fallback tip if AI generation fails"""
        
        fallback_tips = {
            'resume_improvement': [
                f"Quantify your achievements with specific metrics relevant to {context['job_title']}",
                f"Add keywords from the job description: {', '.join(context['missing_skills'][:3])}",
                "Use action verbs to describe your accomplishments"
            ],
            'skill_development': [
                f"Start with {context['missing_skills'][0] if context['missing_skills'] else 'core skills'}",
                "Build a portfolio project demonstrating your skills",
                "Join online communities for continuous learning"
            ],
            'interview_preparation': [
                "Prepare STAR examples for behavioral questions",
                f"Research {context['company']} recent news and culture",
                "Practice explaining technical concepts simply"
            ]
        }
        
        tips_list = fallback_tips.get(category, ["Continue improving your profile"])
        tip_text = tips_list[index % len(tips_list)]
        
        return {
            'category': category,
            'tip': tip_text,
            'priority': 'medium',
            'impact': 'medium',
            'effort': 'medium',
            'timeframe': '1 week',
            'specific_to': context['job_title'],
            'resources': []
        }
    
    def _generate_priority_actions(self, context: Dict, 
                                 comparison_result: ComparisonResult) -> List[Dict]:
        """Generate priority actions based on analysis"""
        
        actions = []
        
        # Priority 1: Address major skill gaps
        if context['skill_gap_severity'] == 'high':
            actions.append({
                'action': f"Focus on learning {context['missing_skills'][0]}",
                'priority': 'critical',
                'deadline': '2 weeks',
                'impact': 'high',
                'reason': 'This is the most important missing skill'
            })
        
        # Priority 2: Optimize resume if score is low
        if context['match_score'] < 70:
            actions.append({
                'action': "Update resume with relevant keywords and achievements",
                'priority': 'high',
                'deadline': '3 days',
                'impact': 'high',
                'reason': 'Improve ATS matching and visibility'
            })
        
        # Priority 3: Prepare for interviews if score is good
        if context['match_score'] >= 70:
            actions.append({
                'action': "Prepare role-specific interview answers",
                'priority': 'high',
                'deadline': '1 week',
                'impact': 'medium',
                'reason': 'You have a good chance, prepare well'
            })
        
        return actions

# Initialize the parser
parser = AdvancedLLMParser()

def llm_parse_streaming(text: str, parse_type: str, progress_callback=None, thought_callback=None) -> Iterator[Tuple[Dict, StreamingProgress]]:
    """Streaming version of llm_parse with real-time progress and thoughts"""
    progress = StreamingProgress()
    start_time = time.time()
    
    try:
        # Initialize model if not already loaded
        if not hasattr(llm_parse_streaming, 'model') or not hasattr(llm_parse_streaming, 'tokenizer'):
            if thought_callback:
                thought_callback(AIThought(
                    stage=AIProcessingStage.INITIALIZING,
                    thought="Loading SmolLM2-135M model into memory...",
                    confidence=1.0,
                    tokens_generated=0,
                    processing_time=0
                ))
            
            from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
            
            llm_parse_streaming.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)
            llm_parse_streaming.model = AutoModelForCausalLM.from_pretrained(
                "HuggingFaceTB/SmolLM2-135M",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            llm_parse_streaming.model.eval()
            
            if llm_parse_streaming.tokenizer.pad_token is None:
                llm_parse_streaming.tokenizer.pad_token = llm_parse_streaming.tokenizer.eos_token
        
        # Multi-stage parsing pipeline
        result = {}
        stages = [
            (AIProcessingStage.ANALYZING_FORMAT, _analyze_document_format),
            (AIProcessingStage.EXTRACTING_STRUCTURE, _extract_document_structure),
            (AIProcessingStage.PARSING_CONTENT, _parse_content_fields),
            (AIProcessingStage.VALIDATING_DATA, _validate_extracted_data),
            (AIProcessingStage.REFINING_RESULTS, _refine_results),
            (AIProcessingStage.FINALIZING, _finalize_parsing)
        ]
        
        for i, (stage, stage_func) in enumerate(stages):
            progress.current_stage = stage
            
            if progress_callback:
                progress_callback(progress)
            
            # Execute stage with streaming
            stage_result, stage_thoughts = stage_func(
                text, parse_type, result, 
                llm_parse_streaming.model, 
                llm_parse_streaming.tokenizer,
                thought_callback
            )
            
            # Update progress
            result.update(stage_result)
            progress.partial_result = result
            progress.thoughts.extend(stage_thoughts)
            progress.overall_confidence = _calculate_overall_confidence(progress.thoughts)
            
            # Calculate processing speed
            elapsed = time.time() - start_time
            if elapsed > 0:
                progress.processing_speed = progress.current_tokens / elapsed
            
            # Yield intermediate results
            yield result.copy(), progress
        
    except Exception as e:
        logger.error(f"Streaming parse error: {e}")
        if thought_callback:
            thought_callback(AIThought(
                stage=progress.current_stage,
                thought=f"Error: {str(e)}",
                confidence=0.0,
                tokens_generated=progress.current_tokens,
                processing_time=time.time() - start_time
            ))
        yield _get_fallback_result(parse_type), progress

def _analyze_document_format(text, parse_type, current_result, model, tokenizer, thought_callback):
    """Stage 1: Analyze document format and structure"""
    thoughts = []
    
    prompt = f"""Analyze the format and structure of this {parse_type}.
What type of document is this? What sections does it contain?
Think step by step.

Document preview:
{text[:500]}

Analysis:"""
    
    # Stream tokens
    thought_text = ""
    tokens_generated = 0
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Use TextIteratorStreamer for real-time token streaming
    from transformers import TextIteratorStreamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Run generation in a separate thread
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=150,
        temperature=0.3,
        do_sample=True,
        top_p=0.9,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Collect streamed tokens
    for token in streamer:
        thought_text += token
        tokens_generated += 1
        
        if thought_callback and tokens_generated % 5 == 0:  # Update every 5 tokens
            thought_callback(AIThought(
                stage=AIProcessingStage.ANALYZING_FORMAT,
                thought=thought_text,
                confidence=0.8,
                tokens_generated=tokens_generated,
                processing_time=0,
                current_field="document_format"
            ))
    
    thread.join()
    
    thoughts.append(AIThought(
        stage=AIProcessingStage.ANALYZING_FORMAT,
        thought=thought_text,
        confidence=0.8,
        tokens_generated=tokens_generated,
        processing_time=0
    ))
    
    # Extract format insights
    format_info = {
        "format_analysis": thought_text,
        "document_type": parse_type,
        "confidence": 0.8
    }
    
    return format_info, thoughts

def _extract_document_structure(text, parse_type, current_result, model, tokenizer, thought_callback):
    """Stage 2: Extract document structure and sections"""
    thoughts = []
    
    if parse_type == "resume":
        sections = ["contact", "summary", "experience", "education", "skills"]
    else:
        sections = ["title", "company", "requirements", "responsibilities", "qualifications"]
    
    structure = {}
    
    for section in sections:
        prompt = f"""Find the {section} section in this document.
Extract relevant information. Be precise.

Document:
{text[:1000]}

{section.capitalize()} information:"""
        
        # Stream extraction for each section
        section_text = _stream_generate_with_callback(
            model, tokenizer, prompt, 
            thought_callback, 
            AIProcessingStage.EXTRACTING_STRUCTURE,
            f"Extracting {section}"
        )
        
        structure[section] = section_text
        
        thoughts.append(AIThought(
            stage=AIProcessingStage.EXTRACTING_STRUCTURE,
            thought=f"Extracted {section}: {section_text[:100]}...",
            confidence=0.75,
            tokens_generated=len(tokenizer.encode(section_text)),
            processing_time=0,
            current_field=section
        ))
    
    return structure, thoughts

def _parse_content_fields(text, parse_type, current_result, model, tokenizer, thought_callback):
    """Stage 3: Parse specific content fields with advanced prompt engineering"""
    thoughts = []
    prompt_engineer = AdvancedPromptEngineer()
    
    if parse_type == "job":
        result = {
            "title": "",
            "company": "",
            "location": "",
            "type": "Full-time",
            "responsibilities": [],
            "required_skills": [],
            "preferred_skills": [],
            "education": "",
            "experience": ""
        }
        
        # Use Chain-of-Thought for each field
        fields = [
            ("title", "What is the exact job title?"),
            ("company", "What company is hiring?"),
            ("location", "Where is this job located?"),
            ("experience", "How many years of experience required?")
        ]
        
        for field, question in fields:
            # Create CoT prompt
            cot_prompt = prompt_engineer.create_chain_of_thought_prompt(
                current_result.get('structure', {}).get(field, text[:500]),
                f"Extract {field}: {question}"
            )
            
            value = _stream_generate_with_callback(
                model, tokenizer, cot_prompt,
                thought_callback,
                AIProcessingStage.PARSING_CONTENT,
                f"Parsing {field} with Chain-of-Thought"
            )
            
            result[field] = value.strip()
            
            thoughts.append(AIThought(
                stage=AIProcessingStage.PARSING_CONTENT,
                thought=f"Parsed {field}: {value} using CoT reasoning",
                confidence=0.85,
                tokens_generated=len(tokenizer.encode(value)),
                processing_time=0,
                current_field=field
            ))
        
        # Extract skills with self-consistency checking
        skill_prompts = prompt_engineer.create_self_consistency_prompts(
            text[:800],
            "required technical skills",
            variations=3
        )
        
        skill_responses = []
        for i, prompt in enumerate(skill_prompts):
            response = _stream_generate_with_callback(
                model, tokenizer, prompt,
                thought_callback,
                AIProcessingStage.PARSING_CONTENT,
                f"Extracting skills (variation {i+1})"
            )
            skill_responses.append(response)
        
        # Aggregate skills from multiple responses
        all_skills = []
        for response in skill_responses:
            skills = _parse_list_response(response)
            all_skills.extend(skills)
        
        # Use most common skills (consensus)
        skill_counts = Counter([s.lower() for s in all_skills])
        result['required_skills'] = [skill for skill, count in skill_counts.most_common(15) if count >= 2]
        
        thoughts.append(AIThought(
            stage=AIProcessingStage.PARSING_CONTENT,
            thought=f"Extracted {len(result['required_skills'])} skills using self-consistency checking",
            confidence=0.9,
            tokens_generated=sum(len(tokenizer.encode(r)) for r in skill_responses),
            processing_time=0,
            current_field="skills"
        ))
        
    else:  # resume
        result = {
            "name": "",
            "email": "",
            "phone": "",
            "summary": "",
            "experience": [],
            "skills": [],
            "education": ""
        }
        
        # Use Chain-of-Thought for complex extractions
        name_cot_prompt = prompt_engineer.create_chain_of_thought_prompt(
            text[:300],
            "Extract the person's full name from the resume"
        )
        
        result['name'] = _stream_generate_with_callback(
            model, tokenizer, name_cot_prompt,
            thought_callback,
            AIProcessingStage.PARSING_CONTENT,
            "Extracting name with reasoning"
        ).strip()
        
        # Skills extraction with multiple approaches
        skill_prompts = prompt_engineer.create_self_consistency_prompts(
            text[:1000],
            "all technical and professional skills",
            variations=3
        )
        
        skill_responses = []
        for i, prompt in enumerate(skill_prompts):
            response = _stream_generate_with_callback(
                model, tokenizer, prompt,
                thought_callback,
                AIProcessingStage.PARSING_CONTENT,
                f"Extracting skills (approach {i+1})"
            )
            skill_responses.append(response)
        
        # Aggregate and deduplicate skills
        all_skills = []
        for response in skill_responses:
            skills = _parse_list_response(response)
            all_skills.extend(skills)
        
        skill_counts = Counter([s.lower() for s in all_skills])
        result['skills'] = [skill for skill, count in skill_counts.most_common(20) if count >= 2]
        
        # Use regex for email/phone as fallback
        import re
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        if email_match:
            result['email'] = email_match.group(0)
        
        phone_match = re.search(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text)
        if phone_match:
            result['phone'] = phone_match.group(0)
    
    return result, thoughts

def _validate_extracted_data(text, parse_type, current_result, model, tokenizer, thought_callback):
    """Stage 4: Validate extracted data"""
    thoughts = []
    
    validation_prompt = f"""Review this extracted {parse_type} data for accuracy and completeness.
What might be missing or incorrect?

Extracted data:
{json.dumps(current_result, indent=2)}

Original text preview:
{text[:500]}

Validation analysis:"""
    
    validation_text = _stream_generate_with_callback(
        model, tokenizer, validation_prompt,
        thought_callback,
        AIProcessingStage.VALIDATING_DATA,
        "Validating extracted data"
    )
    
    thoughts.append(AIThought(
        stage=AIProcessingStage.VALIDATING_DATA,
        thought=validation_text,
        confidence=0.85,
        tokens_generated=len(tokenizer.encode(validation_text)),
        processing_time=0
    ))
    
    # Apply corrections based on validation
    corrections = {}
    if "missing" in validation_text.lower():
        # AI identified missing fields
        missing_prompt = f"""Fill in any missing critical fields based on the document.

Current data:
{json.dumps(current_result, indent=2)}

Document:
{text[:800]}

Missing fields with values:"""
        
        corrections_text = _stream_generate_with_callback(
            model, tokenizer, missing_prompt,
            thought_callback,
            AIProcessingStage.VALIDATING_DATA,
            "Filling missing fields"
        )
        
        # Parse corrections (simplified)
        for line in corrections_text.split('\n'):
            if ':' in line:
                field, value = line.split(':', 1)
                field = field.strip().lower().replace(' ', '_')
                if field in current_result and not current_result.get(field):
                    corrections[field] = value.strip()
    
    return corrections, thoughts

def _refine_results(text, parse_type, current_result, model, tokenizer, thought_callback):
    """Stage 5: Refine and enhance results"""
    thoughts = []
    refinements = {}
    
    # Enhance skills extraction
    if parse_type == "job" and current_result.get('required_skills'):
        skills_enhancement_prompt = f"""Review these extracted skills and identify any implied or related skills.
Also categorize them by type (programming, frameworks, tools, soft skills).

Current skills: {', '.join(current_result.get('required_skills', []))}

Enhanced skill analysis:"""
        
        enhanced_skills = _stream_generate_with_callback(
            model, tokenizer, skills_enhancement_prompt,
            thought_callback,
            AIProcessingStage.REFINING_RESULTS,
            "Enhancing skills analysis"
        )
        
        thoughts.append(AIThought(
            stage=AIProcessingStage.REFINING_RESULTS,
            thought=f"Enhanced skills analysis: {enhanced_skills[:200]}...",
            confidence=0.9,
            tokens_generated=len(tokenizer.encode(enhanced_skills)),
            processing_time=0,
            current_field="skills_enhancement"
        ))
        
        refinements['skills_analysis'] = enhanced_skills
    
    # Add quality score
    quality_prompt = f"""Rate the quality and completeness of this {parse_type} from 0-100.
Consider: clarity, detail level, professional presentation.

Brief explanation and score:"""
    
    quality_assessment = _stream_generate_with_callback(
        model, tokenizer, quality_prompt,
        thought_callback,
        AIProcessingStage.REFINING_RESULTS,
        "Assessing document quality"
    )
    
    refinements['quality_score'] = quality_assessment
    
    return refinements, thoughts

def _finalize_parsing(text, parse_type, current_result, model, tokenizer, thought_callback):
    """Stage 6: Finalize parsing results"""
    thoughts = []
    
    # Generate final summary
    summary_prompt = f"""Provide a brief summary of this {parse_type} parsing result.
What are the key takeaways?

Summary:"""
    
    summary = _stream_generate_with_callback(
        model, tokenizer, summary_prompt,
        thought_callback,
        AIProcessingStage.FINALIZING,
        "Generating final summary"
    )
    
    thoughts.append(AIThought(
        stage=AIProcessingStage.FINALIZING,
        thought=f"Parsing complete. Summary: {summary}",
        confidence=0.95,
        tokens_generated=len(tokenizer.encode(summary)),
        processing_time=0
    ))
    
    final_additions = {
        'parsing_summary': summary,
        'parsing_timestamp': datetime.now().isoformat(),
        'model_version': 'SmolLM2-135M'
    }
    
    return final_additions, thoughts

def _stream_generate_with_callback(model, tokenizer, prompt, thought_callback, stage, field_name):
    """Helper function for streaming generation with callbacks"""
    from transformers import TextIteratorStreamer
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    generated_text = ""
    tokens = 0
    
    for token in streamer:
        generated_text += token
        tokens += 1
        
        if thought_callback and tokens % 3 == 0:
            thought_callback(AIThought(
                stage=stage,
                thought=f"Generating {field_name}: {generated_text}",
                confidence=0.8,
                tokens_generated=tokens,
                processing_time=0,
                current_field=field_name
            ))
    
    thread.join()
    return generated_text

def _calculate_overall_confidence(thoughts: List[AIThought]) -> float:
    """Calculate overall confidence from all thoughts"""
    if not thoughts:
        return 0.0
    confidences = [t.confidence for t in thoughts if t.confidence > 0]
    return sum(confidences) / len(confidences) if confidences else 0.0

def _get_fallback_result(parse_type: str) -> Dict:
    """Return minimal structure if AI fails"""
    if parse_type == "job":
        return {
            "title": "Could not parse job title",
            "company": "",
            "location": "",
            "type": "Full-time",
            "responsibilities": [],
            "required_skills": [],
            "preferred_skills": [],
            "education": "",
            "experience": ""
        }
    else:
        return {
            "name": "Could not parse name",
            "email": "",
            "phone": "",
            "summary": "",
            "experience": [],
            "skills": [],
            "education": ""
        }

# Keep the old llm_parse function for backward compatibility
def llm_parse(text: str, parse_type: str, model: str = "SmolLM2-135M") -> Dict:
    """Backward compatible wrapper using streaming parser"""
    result = {}
    for partial_result, progress in llm_parse_streaming(text, parse_type):
        result = partial_result
    return result

def _generate_single_response(model, tokenizer, prompt, max_tokens=50):
    """Generate a single response from the model"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the generated part
    response = response[len(prompt):].strip()
    
    # Clean up response
    response = response.split('\n')[0].strip()  # Take first line only
    return response

def _parse_list_response(response):
    """Parse a list response into individual items"""
    items = []
    lines = response.split('\n')
    for line in lines:
        # Remove numbering, bullets, etc.
        cleaned = re.sub(r'^[\d\.\-\*\s]+', '', line).strip()
        if cleaned and len(cleaned) > 1:
            items.append(cleaned)
    return items[:10]  # Limit to 10 items

def enrich_parsed_data(parsed_data: Dict, parse_type: str, original_text: str) -> Dict:
    """Enrich parsed data by filling in missing fields"""
    
    if parse_type == "job":
        # If skills are empty, try to extract from the full text
        if not parsed_data.get('required_skills'):
            # Look for any technical terms in the text
            tech_terms = re.findall(
                r'\b(?:Python|Java|JavaScript|React|Node|SQL|AWS|Docker|Git|API|REST|Agile|Scrum)\b',
                original_text, re.I
            )
            parsed_data['required_skills'] = list(set(tech_terms))
        
        # If no location, default to "Not specified"
        if not parsed_data.get('location'):
            parsed_data['location'] = "Location not specified"
        
        # If no experience, look harder
        if not parsed_data.get('experience'):
            exp_match = re.search(r'(\d+).*?year', original_text, re.I)
            if exp_match:
                parsed_data['experience'] = f"{exp_match.group(1)}+ years"
            else:
                parsed_data['experience'] = "Experience not specified"
        
        # If no education, check for degree mentions
        if not parsed_data.get('education'):
            if re.search(r'bachelor|degree|BS|BA', original_text, re.I):
                parsed_data['education'] = "Bachelor's degree required"
            else:
                parsed_data['education'] = "Education not specified"
                
    else:  # resume
        # Ensure phone has a default
        if not parsed_data.get('phone'):
            parsed_data['phone'] = "Phone not provided"
        
        # Ensure education has a value
        if not parsed_data.get('education'):
            parsed_data['education'] = "Education not specified"
        
        # Ensure at least some experience
        if not parsed_data.get('experience'):
            parsed_data['experience'] = []
    
    return parsed_data

# AI Thinking Visualization Component
def render_ai_thinking_panel(placeholder, progress: StreamingProgress, thoughts_history: List[AIThought]):
    """Render real-time AI thinking panel"""
    with placeholder.container():
        # Progress header
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Create a mapping of stages to indices
            stage_mapping = {
                AIProcessingStage.INITIALIZING: 1,
                AIProcessingStage.ANALYZING_FORMAT: 2,
                AIProcessingStage.EXTRACTING_STRUCTURE: 3,
                AIProcessingStage.PARSING_CONTENT: 4,
                AIProcessingStage.VALIDATING_DATA: 5,
                AIProcessingStage.REFINING_RESULTS: 6,
                AIProcessingStage.FINALIZING: 7
            }
            
            stage_index = stage_mapping.get(progress.current_stage, 1)
            st.progress(stage_index / progress.total_stages)
            st.caption(f"Stage {stage_index}/{progress.total_stages}: {progress.current_stage.value}")
        
        with col2:
            if progress.processing_speed > 0:
                st.metric("Speed", f"{progress.processing_speed:.1f} tok/s")
            else:
                st.metric("Speed", "-- tok/s")
        
        with col3:
            st.metric("Confidence", f"{progress.overall_confidence*100:.1f}%")
        
        # Current thinking
        if thoughts_history:
            latest_thought = thoughts_history[-1]
            
            # Thinking animation
            thinking_html = f"""
            <div style="
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                padding: 2px;
                border-radius: 10px;
                margin: 10px 0;
            ">
                <div style="
                    background: rgba(255, 255, 255, 0.95);
                    padding: 15px;
                    border-radius: 8px;
                ">
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <div style="
                            width: 10px;
                            height: 10px;
                            background: #667eea;
                            border-radius: 50%;
                            margin-right: 10px;
                            animation: pulse 1s ease-in-out infinite;
                        "></div>
                        <strong>AI is thinking about: {latest_thought.current_field}</strong>
                    </div>
                    <div style="
                        font-family: 'Courier New', monospace;
                        font-size: 14px;
                        color: #333;
                        white-space: pre-wrap;
                        max-height: 150px;
                        overflow-y: auto;
                        background: #f8f9fa;
                        padding: 10px;
                        border-radius: 5px;
                    ">
                        {latest_thought.thought}
                    </div>
                    <div style="
                        margin-top: 10px;
                        font-size: 12px;
                        color: #666;
                    ">
                        Tokens generated: {latest_thought.tokens_generated} | 
                        Confidence: {latest_thought.confidence*100:.1f}%
                    </div>
                </div>
            </div>
            <style>
                @keyframes pulse {{
                    0%, 100% {{ opacity: 1; }}
                    50% {{ opacity: 0.5; }}
                }}
            </style>
            """
            st.markdown(thinking_html, unsafe_allow_html=True)
        
        # Stage history
        with st.expander("Processing History", expanded=False):
            for thought in thoughts_history[-10:]:  # Last 10 thoughts
                st.caption(f"**{thought.stage.value}** - {thought.current_field}")
                st.text(thought.thought[:100] + "..." if len(thought.thought) > 100 else thought.thought)
                st.caption(f"Confidence: {thought.confidence*100:.1f}% | Tokens: {thought.tokens_generated}")
                st.divider()

# Comparison engine
def calculate_match_score(job: Dict, resume: Dict) -> Tuple[float, ComparisonResult]:
    """Calculate match score using AI model"""
    
    # Initialize model if needed
    if not hasattr(calculate_match_score, 'model') or not hasattr(calculate_match_score, 'tokenizer'):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        calculate_match_score.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)
        calculate_match_score.model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        calculate_match_score.model.eval()
        
        if calculate_match_score.tokenizer.pad_token is None:
            calculate_match_score.tokenizer.pad_token = calculate_match_score.tokenizer.eos_token
    
    # Prepare job and resume summaries for AI
    job_summary = f"""
Job Title: {job.get('title', 'Not specified')}
Company: {job.get('company', 'Not specified')}
Skills Required: {', '.join(job.get('required_skills', [])) or 'Not specified'}
Experience: {job.get('experience', 'Not specified')}
Education: {job.get('education', 'Not specified')}
"""

    resume_summary = f"""
Name: {resume.get('name', 'Not specified')}
Skills: {', '.join(resume.get('skills', [])) or 'Not specified'}
Education: {resume.get('education', 'Not specified')}
"""

    # AI Score Calculation
    score_prompt = f"""Rate how well this resume matches the job on a scale of 0-100.
Consider skills match, experience, and education. Just give a number.

Job Requirements:
{job_summary}

Candidate Resume:
{resume_summary}

Match score (0-100):"""

    score_response = _generate_single_response(
        calculate_match_score.model, 
        calculate_match_score.tokenizer, 
        score_prompt, 
        max_tokens=10
    )
    
    # Extract score
    try:
        score = float(re.findall(r'\d+', score_response)[0])
        score = min(100, max(0, score))
    except:
        score = 50.0  # Default if AI fails
    
    # AI Skills Analysis
    skills_prompt = f"""Compare the candidate's skills with job requirements.

Job needs: {', '.join(job.get('required_skills', [])) or 'Any skills'}
Candidate has: {', '.join(resume.get('skills', [])) or 'No skills listed'}

List matching skills and missing skills:"""

    skills_response = _generate_single_response(
        calculate_match_score.model,
        calculate_match_score.tokenizer,
        skills_prompt,
        max_tokens=100
    )
    
    # Parse skills response
    matched_skills = []
    missing_skills = []
    
    # Try to extract from AI response
    if 'match' in skills_response.lower():
        matches = re.findall(r'match[:\s]+([^,\n]+)', skills_response, re.I)
        if matches:
            matched_skills = [s.strip() for s in matches[0].split(',')]
    
    if 'miss' in skills_response.lower():
        misses = re.findall(r'miss[:\s]+([^,\n]+)', skills_response, re.I)
        if misses:
            missing_skills = [s.strip() for s in misses[0].split(',')]
    
    # Fallback skill matching if AI doesn't provide clear answer
    if not matched_skills and not missing_skills:
        job_skills = set([s.lower() for s in job.get('required_skills', [])])
        resume_skills = set([s.lower() for s in resume.get('skills', [])])
        matched_skills = list(job_skills.intersection(resume_skills))
        missing_skills = list(job_skills - resume_skills)
    
    # AI Experience Analysis
    exp_prompt = f"""Does the candidate meet the experience requirement?

Job requires: {job.get('experience', 'No specific requirement')}
Candidate info: {resume.get('raw_text', '')[:500]}

Answer briefly:"""

    exp_response = _generate_single_response(
        calculate_match_score.model,
        calculate_match_score.tokenizer,
        exp_prompt,
        max_tokens=50
    )
    
    exp_status = exp_response.strip() or "Experience assessment unclear"
    
    # AI Education Analysis
    edu_prompt = f"""Does the candidate's education match the job requirement?

Job requires: {job.get('education', 'No specific requirement')}
Candidate has: {resume.get('education', 'Not specified')}

Answer briefly:"""

    edu_response = _generate_single_response(
        calculate_match_score.model,
        calculate_match_score.tokenizer,
        edu_prompt,
        max_tokens=50
    )
    
    edu_status = edu_response.strip() or "Education assessment unclear"
    
    # AI Recommendations
    rec_prompt = f"""Based on this match score of {score}%, what should the candidate do to improve?
Give 3 short recommendations:

1."""

    rec_response = _generate_single_response(
        calculate_match_score.model,
        calculate_match_score.tokenizer,
        rec_prompt,
        max_tokens=150
    )
    
    # Parse recommendations
    recommendations = []
    rec_lines = rec_response.split('\n')
    for line in rec_lines:
        line = re.sub(r'^\d+\.?\s*', '', line).strip()
        if line and len(line) > 5:
            recommendations.append(line)
    
    # If no good recommendations from AI, provide basic ones
    if len(recommendations) < 2:
        if score < 60:
            recommendations = [
                "Develop the missing skills identified",
                "Gain more relevant experience",
                "Highlight transferable skills in your resume"
            ]
        elif score < 80:
            recommendations = [
                "Focus on the few missing skills",
                "Emphasize your matching qualifications",
                "Tailor your resume to the job description"
            ]
        else:
            recommendations = [
                "Your profile is a great match!",
                "Emphasize your key strengths in the interview",
                "Research the company culture and values"
            ]
    
    # AI Overall Summary
    summary_prompt = f"""In one sentence, summarize this job match with {score}% score:"""
    
    summary_response = _generate_single_response(
        calculate_match_score.model,
        calculate_match_score.tokenizer,
        summary_prompt,
        max_tokens=50
    )
    
    overall_summary = summary_response.strip() or f"{'Excellent' if score >= 80 else 'Good' if score >= 60 else 'Fair'} match for this position"
    
    # Build result
    result = ComparisonResult(
        match_score=round(score, 1),
        overall_summary=overall_summary,
        skill_matches_detailed={
            "matched": matched_skills,
            "missing": missing_skills,
            "score": round(score, 1)
        },
        experience_match_status=exp_status,
        education_match_status=edu_status,
        ai_recommendations=recommendations[:3],  # Top 3 recommendations
        missing_keywords=missing_skills[:5],
        ai_confidence_scores={
            "skills": round(score, 1),
            "experience": round(score, 1),
            "education": round(score, 1),
            "overall": round(score, 1)
        }
    )
    
    return score, result

# Recommendation Engine
class RecommendationEngine:
    def __init__(self):
        self.skill_relationships = {
            'python': ['django', 'flask', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch'],
            'java': ['spring', 'hibernate', 'maven', 'gradle', 'junit', 'mockito'],
            'javascript': ['react', 'vue', 'angular', 'node.js', 'express', 'typescript'],
            'frontend': ['react', 'vue', 'angular', 'typescript', 'html', 'css', 'sass'],
            'backend': ['node.js', 'express', 'django', 'flask', 'spring', 'fastapi'],
            'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch'],
            'cloud': ['aws', 'azure', 'gcp', 'kubernetes', 'docker', 'terraform'],
            'devops': ['jenkins', 'gitlab', 'github actions', 'docker', 'kubernetes'],
            'data': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'spark'],
            'mobile': ['react native', 'flutter', 'swift', 'kotlin', 'android', 'ios']
        }
        
        self.learning_resources = {
            'python': {
                'beginner': ['Python Crash Course on Coursera', 'Python.org tutorials', 'Codecademy Python'],
                'intermediate': ['Real Python', 'Python Design Patterns', 'Effective Python'],
                'advanced': ['Fluent Python', 'High Performance Python', 'Python Cookbook']
            },
            'java': {
                'beginner': ['Java Tutorial on Oracle', 'Head First Java', 'Codecademy Java'],
                'intermediate': ['Effective Java', 'Clean Code', 'Java Concurrency in Practice'],
                'advanced': ['Java Performance', 'Design Patterns', 'Spring in Action']
            },
            'javascript': {
                'beginner': ['JavaScript.info', 'Eloquent JavaScript', 'You Don\'t Know JS'],
                'intermediate': ['JavaScript: The Good Parts', 'Functional Programming in JS'],
                'advanced': ['JavaScript Patterns', 'High Performance JavaScript']
            },
            'react': {
                'beginner': ['React Official Tutorial', 'React for Beginners', 'Codecademy React'],
                'intermediate': ['React Design Patterns', 'Advanced React Patterns'],
                'advanced': ['React Performance', 'Testing React Applications']
            },
            'aws': {
                'beginner': ['AWS Free Tier', 'AWS Fundamentals', 'AWS Cloud Practitioner'],
                'intermediate': ['AWS Solutions Architect', 'AWS Developer Associate'],
                'advanced': ['AWS Solutions Architect Professional', 'AWS DevOps Professional']
            },
            'data_science': {
                'beginner': ['Data Science Handbook', 'Python for Data Analysis', 'Kaggle Courses'],
                'intermediate': ['Introduction to Statistical Learning', 'Hands-On Machine Learning'],
                'advanced': ['Elements of Statistical Learning', 'Deep Learning by Ian Goodfellow']
            }
        }
    
    def generate_smart_recommendations(self, job: Dict, resume: Dict, comparison: ComparisonResult) -> Dict:
        """Generate recommendations using AI"""
        
        # Initialize model if needed
        if not hasattr(self, 'model') or not hasattr(self, 'tokenizer'):
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                "HuggingFaceTB/SmolLM2-135M",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            self.model.eval()
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        recommendations = {
            'immediate_actions': [],
            'skill_roadmap': [],
            'resume_optimizations': [],
            'estimated_timeline': '',
            'success_probability': comparison.match_score
        }
        
        # AI: Immediate Actions
        action_prompt = f"""Based on a {comparison.match_score}% job match, what are 3 immediate actions the candidate should take?

Missing skills: {', '.join(comparison.skill_matches_detailed['missing'][:5]) or 'None'}

Actions:
1."""

        action_response = _generate_single_response(self.model, self.tokenizer, action_prompt, max_tokens=150)
        
        actions = []
        for line in action_response.split('\n'):
            line = re.sub(r'^\d+\.?\s*', '', line).strip()
            if line and len(line) > 5:
                actions.append(line)
        
        recommendations['immediate_actions'] = actions[:3]
        
        # AI: Skill Roadmap
        if comparison.skill_matches_detailed['missing']:
            for skill in comparison.skill_matches_detailed['missing'][:5]:
                skill_prompt = f"""How difficult is it to learn {skill} and how long will it take?
Current skills: {', '.join(resume.get('skills', [])[:10])}

Answer with: difficulty (Easy/Medium/Hard) and time estimate:"""

                skill_response = _generate_single_response(self.model, self.tokenizer, skill_prompt, max_tokens=50)
                
                # Parse response
                difficulty = "Medium"
                time_estimate = "2-4 weeks"
                
                if 'easy' in skill_response.lower():
                    difficulty = "Easy"
                    time_estimate = "1-2 weeks"
                elif 'hard' in skill_response.lower() or 'difficult' in skill_response.lower():
                    difficulty = "Hard"
                    time_estimate = "2-3 months"
                
                # Extract time if mentioned
                time_match = re.search(r'(\d+[-\s]\d+\s*(?:week|month))', skill_response, re.I)
                if time_match:
                    time_estimate = time_match.group(1)
                
                recommendations['skill_roadmap'].append({
                    'skill': skill,
                    'foundation': None,  # AI could determine this too if needed
                    'difficulty': difficulty,
                    'time_estimate': time_estimate,
                    'resources': [f"Learn {skill} online", f"{skill} tutorials", f"{skill} documentation"]
                })
        
        # AI: Resume Optimizations
        resume_prompt = f"""How can this resume be improved for a {comparison.match_score}% match?

Job needs: {', '.join(job.get('required_skills', [])[:5])}
Candidate has: {', '.join(resume.get('skills', [])[:5])}

Give 2 specific resume improvements:
1."""

        resume_response = _generate_single_response(self.model, self.tokenizer, resume_prompt, max_tokens=150)
        
        improvements = []
        for i, line in enumerate(resume_response.split('\n')[:2]):
            line = re.sub(r'^\d+\.?\s*', '', line).strip()
            if line and len(line) > 5:
                improvements.append({
                    'type': 'general',
                    'priority': 'High' if i == 0 else 'Medium',
                    'suggestion': line,
                    'example': f"Update your resume to: {line[:50]}..."
                })
        
        recommendations['resume_optimizations'] = improvements
        
        # AI: Timeline Estimate
        timeline_prompt = f"""How long to improve from {comparison.match_score}% to 80%+ match?
Need to learn: {len(comparison.skill_matches_detailed['missing'])} skills

Timeline estimate:"""

        timeline_response = _generate_single_response(self.model, self.tokenizer, timeline_prompt, max_tokens=30)
        
        # Extract timeline
        if 'month' in timeline_response.lower():
            recommendations['estimated_timeline'] = re.findall(r'\d+[-\s]\d+\s*month', timeline_response)[0] if re.findall(r'\d+[-\s]\d+\s*month', timeline_response) else "2-3 months"
        elif 'week' in timeline_response.lower():
            recommendations['estimated_timeline'] = re.findall(r'\d+[-\s]\d+\s*week', timeline_response)[0] if re.findall(r'\d+[-\s]\d+\s*week', timeline_response) else "4-6 weeks"
        else:
            recommendations['estimated_timeline'] = "1-3 months"
        
        # AI: Success Probability
        prob_prompt = f"""What's the probability of success if the candidate follows all recommendations?
Current match: {comparison.match_score}%

Success probability (0-100):"""

        prob_response = _generate_single_response(self.model, self.tokenizer, prob_prompt, max_tokens=10)
        
        try:
            prob = float(re.findall(r'\d+', prob_response)[0])
            recommendations['success_probability'] = min(100, max(0, prob))
        except:
            recommendations['success_probability'] = min(95, comparison.match_score + 20)
        
        return recommendations
    
    def _find_skill_foundation(self, current_skills: List[str], missing_skills: List[str]) -> Dict[str, List[str]]:
        """Find which current skills can serve as foundation for missing skills"""
        foundation = {}
        current_skills_lower = [s.lower() for s in current_skills]
        
        for missing in missing_skills:
            missing_lower = missing.lower()
            related = []
            
            for category, skills in self.skill_relationships.items():
                if missing_lower in [s.lower() for s in skills]:
                    # Find which skills from this category they already have
                    for skill in skills:
                        if skill.lower() in current_skills_lower:
                            related.append(skill)
            
            if related:
                foundation[missing] = related
        
        return foundation
    
    def _get_related_skills(self, target_skill: str, current_skills: List[str]) -> List[str]:
        """Get skills related to target that user already has"""
        target_lower = target_skill.lower()
        current_lower = [s.lower() for s in current_skills]
        related = []
        
        for category, skills in self.skill_relationships.items():
            if target_lower in [s.lower() for s in skills]:
                for skill in skills:
                    if skill.lower() in current_lower:
                        related.append(skill)
        
        return related
    
    def _get_learning_resources(self, skill: str) -> List[str]:
        """Get learning resources for a skill"""
        skill_lower = skill.lower()
        
        # Map skill to category
        for category, resources in self.learning_resources.items():
            if category in skill_lower or skill_lower in category:
                return resources.get('beginner', []) + resources.get('intermediate', [])
        
        # Default resources
        return [
            f"{skill} Official Documentation",
            f"{skill} Tutorial on YouTube",
            f"Online Course: {skill} for Beginners"
        ]
    
    def _analyze_resume_improvements(self, job: Dict, resume: Dict, comparison: ComparisonResult) -> List[Dict]:
        """Suggest specific resume improvements"""
        improvements = []
        
        # Keyword analysis
        job_text = job.get('raw_text', '').lower()
        resume_text = resume.get('raw_text', '').lower()
        
        # Find important keywords missing from resume
        important_keywords = self._extract_important_keywords(job_text)
        missing_keywords = [kw for kw in important_keywords if kw not in resume_text]
        
        if missing_keywords:
            improvements.append({
                'type': 'keywords',
                'priority': 'High',
                'suggestion': f"Add these keywords naturally: {', '.join(missing_keywords[:5])}",
                'example': self._generate_keyword_example(missing_keywords[0])
            })
        
        # Structure improvements
        if len(resume.get('experience', [])) < 3:
            improvements.append({
                'type': 'structure',
                'priority': 'Medium',
                'suggestion': "Expand work experience section with quantifiable achievements",
                'example': "Increased system performance by 40% through optimization..."
            })
        
        # Skills section improvements
        if len(resume.get('skills', [])) < 10:
            improvements.append({
                'type': 'skills',
                'priority': 'Medium',
                'suggestion': "Expand skills section with specific technologies and tools",
                'example': "Add specific versions, frameworks, and tools you've used"
            })
        
        return improvements
    
    def _extract_important_keywords(self, job_text: str) -> List[str]:
        """Extract important keywords from job description"""
        # Simple keyword extraction - in production, use NLP
        keywords = []
        common_tech = ['python', 'java', 'javascript', 'react', 'aws', 'docker', 'kubernetes', 
                      'sql', 'mongodb', 'node.js', 'spring', 'django', 'flask', 'git']
        
        for keyword in common_tech:
            if keyword in job_text:
                keywords.append(keyword)
        
        return keywords
    
    def _generate_keyword_example(self, keyword: str) -> str:
        """Generate example of how to use keyword in resume"""
        examples = {
            'python': "Developed Python scripts to automate data processing, reducing manual work by 60%",
            'react': "Built responsive web applications using React, improving user engagement by 25%",
            'aws': "Deployed applications on AWS, reducing infrastructure costs by 30%",
            'docker': "Containerized applications using Docker, improving deployment consistency",
            'sql': "Optimized SQL queries, reducing database response time by 40%"
        }
        return examples.get(keyword.lower(), f"Used {keyword} to improve system performance")
    
    def _calculate_success_probability(self, comparison: ComparisonResult, skill_foundation: Dict) -> float:
        """Calculate probability of success based on current match and skill foundation"""
        base_score = comparison.match_score
        
        # Bonus for having related skills
        foundation_bonus = len(skill_foundation) * 5  # 5% bonus per related skill
        
        # Bonus for high confidence scores
        confidence_bonus = sum(comparison.ai_confidence_scores.values()) / len(comparison.ai_confidence_scores) * 10
        
        final_probability = min(100, base_score + foundation_bonus + confidence_bonus)
        return final_probability
    
    def _generate_immediate_actions(self, job: Dict, resume: Dict, comparison: ComparisonResult) -> List[str]:
        """Generate immediate actionable steps"""
        actions = []
        
        # Quick wins
        if comparison.match_score < 60:
            actions.append("Focus on the top 2-3 missing skills first")
        
        if len(resume.get('skills', [])) < 15:
            actions.append("Add more specific technical skills to your resume")
        
        if not resume.get('summary'):
            actions.append("Add a compelling professional summary")
        
        # Research actions
        actions.append(f"Research {job.get('company', 'the company')} culture and values")
        actions.append("Practice common interview questions for this role")
        
        return actions
    
    def _estimate_timeline(self, skill_roadmap: List[Dict]) -> str:
        """Estimate timeline for skill development"""
        if not skill_roadmap:
            return "No skills to develop"
        
        total_weeks = sum(
            DIFFICULTY_WEEKS.get(skill.get('difficulty'), DEFAULT_WEEKS_PER_SKILL)
            for skill in skill_roadmap
        )
        
        if total_weeks <= 4:
            return f"{total_weeks} weeks"
        elif total_weeks <= 12:
            months = total_weeks // 4
            return f"{months} months"
        else:
            return f"{total_weeks // 4} months"

# Initialize recommendation engine
rec_engine = RecommendationEngine()

# Sharing Engine
class SharingEngine:
    def __init__(self):
        self.share_links = {}  # In production, use Redis
        
    def create_shareable_link(self, comparison_id: str, expires_hours: int = 48) -> str:
        """Create secure shareable link"""
        token = secrets.token_urlsafe(16)
        
        self.share_links[token] = {
            'comparison_id': comparison_id,
            'created': datetime.now(),
            'expires': datetime.now() + timedelta(hours=expires_hours),
            'views': 0
        }
        
        # In production, use your actual domain
        share_url = f"http://localhost:8501/?share={token}"
        return share_url, token
    
    def generate_qr_code(self, url: str) -> str:
        """Generate QR code for easy sharing"""
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(url)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buf = BytesIO()
        img.save(buf, format='PNG')
        
        return base64.b64encode(buf.getvalue()).decode()
    
    def create_professional_report(self, comparison: Dict) -> bytes:
        """Generate downloadable PDF report"""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
            
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            story.append(Paragraph(f"Job Match Analysis Report", styles['Title']))
            story.append(Spacer(1, 12))
            
            # Summary
            story.append(Paragraph(f"Candidate: {comparison.get('candidate_name', 'Unknown')}", styles['Heading2']))
            story.append(Paragraph(f"Position: {comparison.get('job_title', 'Unknown')}", styles['Heading2']))
            story.append(Paragraph(f"Match Score: {comparison.get('match_score', 0)}%", styles['Heading1']))
            story.append(Spacer(1, 12))
            
            # Detailed analysis
            comparison_results = comparison.get('comparison_results', {})
            if comparison_results:
                story.append(Paragraph("Detailed Analysis", styles['Heading2']))
                story.append(Spacer(1, 12))
                
                # Skills analysis
                skill_details = comparison_results.get('skill_matches_detailed', {})
                if skill_details:
                    story.append(Paragraph(f"Skills Match: {skill_details.get('score', 0):.1f}%", styles['Normal']))
                    
                    if skill_details.get('matched'):
                        story.append(Paragraph("Matched Skills:", styles['Heading3']))
                        for skill in skill_details['matched'][:10]:
                            story.append(Paragraph(f"• {skill}", styles['Normal']))
                    
                    if skill_details.get('missing'):
                        story.append(Paragraph("Missing Skills:", styles['Heading3']))
                        for skill in skill_details['missing'][:10]:
                            story.append(Paragraph(f"• {skill}", styles['Normal']))
                
                # Experience and education
                story.append(Paragraph(f"Experience: {comparison_results.get('experience_match_status', '')}", styles['Normal']))
                story.append(Paragraph(f"Education: {comparison_results.get('education_match_status', '')}", styles['Normal']))
                
                # Recommendations
                if comparison_results.get('ai_recommendations'):
                    story.append(Paragraph("AI Recommendations:", styles['Heading3']))
                    for rec in comparison_results.get('ai_recommendations', []):
                        story.append(Paragraph(f"• {rec}", styles['Normal']))
            
            # Generate smart recommendations if possible
            if comparison.get('parsed_data'):
                try:
                    job_data = comparison['parsed_data'].get('job', {})
                    resume_data = comparison['parsed_data'].get('resume', {})
                    
                    # Create ComparisonResult object for recommendations
                    results_obj = ComparisonResult(
                        match_score=comparison.get('match_score', 0),
                        overall_summary=comparison_results.get('overall_summary', ''),
                        skill_matches_detailed=comparison_results.get('skill_matches_detailed', {}),
                        experience_match_status=comparison_results.get('experience_match_status', ''),
                        education_match_status=comparison_results.get('education_match_status', ''),
                        ai_recommendations=comparison_results.get('ai_recommendations', []),
                        missing_keywords=comparison_results.get('missing_keywords', []),
                        ai_confidence_scores=comparison_results.get('ai_confidence_scores', {})
                    )
                    
                    smart_recs = rec_engine.generate_smart_recommendations(job_data, resume_data, results_obj)
                    
                    # Add recommendations to report
                    story.append(Spacer(1, 12))
                    story.append(Paragraph("Smart Recommendations", styles['Heading2']))
                    story.append(Paragraph(f"Success Probability: {smart_recs['success_probability']:.1f}%", styles['Normal']))
                    story.append(Paragraph(f"Timeline: {smart_recs['estimated_timeline']}", styles['Normal']))
                    
                    if smart_recs['skill_roadmap']:
                        story.append(Paragraph("Skill Development Roadmap:", styles['Heading3']))
                        for skill in smart_recs['skill_roadmap'][:5]:
                            story.append(Paragraph(f"• {skill['skill']} - {skill['time_estimate']} ({skill['difficulty']})", styles['Normal']))
                    
                except Exception as e:
                    story.append(Paragraph(f"Note: Could not generate detailed recommendations due to: {str(e)}", styles['Normal']))
            
            # Footer
            story.append(Spacer(1, 20))
            story.append(Paragraph(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Paragraph("Generated by Job-Resume Matcher", styles['Normal']))
            
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()
            
        except ImportError:
            # Fallback if reportlab is not available
            buffer = BytesIO()
            report_text = f"""
Job Match Analysis Report

Candidate: {comparison.get('candidate_name', 'Unknown')}
Position: {comparison.get('job_title', 'Unknown')}
Match Score: {comparison.get('match_score', 0)}%

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Generated by Job-Resume Matcher
            """
            buffer.write(report_text.encode())
            buffer.seek(0)
            return buffer.getvalue()
    
    def get_shared_comparison(self, token: str) -> Optional[Dict]:
        """Get shared comparison by token"""
        if token in self.share_links:
            link_data = self.share_links[token]
            
            # Check if expired
            if datetime.now() > link_data['expires']:
                return None
            
            # Increment view count
            link_data['views'] += 1
            
            # Load comparison data
            comparison_file = COMPARISONS_DIR / f"{link_data['comparison_id']}.json"
            if comparison_file.exists():
                with open(comparison_file, 'r') as f:
                    return json.load(f)
        
        return None

# Initialize sharing engine
sharing = SharingEngine()

# Data persistence
def save_job(job: Dict) -> str:
    job_id = job.get('id', str(uuid.uuid4()))
    job['id'] = job_id
    job['date_added'] = job.get('date_added', datetime.now().strftime('%Y-%m-%d'))
    
    with open(JOBS_DIR / f"{job_id}.json", 'w') as f:
        json.dump(job, f, indent=2)
    
    return job_id

def save_resume(resume: Dict) -> str:
    resume_id = resume.get('id', str(uuid.uuid4()))
    resume['id'] = resume_id
    
    with open(RESUMES_DIR / f"{resume_id}.json", 'w') as f:
        json.dump(resume, f, indent=2)
    
    return resume_id

def save_comparison(comparison: Dict) -> str:
    comp_id = str(uuid.uuid4())
    comparison['id'] = comp_id
    comparison['date'] = datetime.now().strftime('%Y-%m-%d')
    comparison['status'] = 'Completed'
    
    with open(COMPARISONS_DIR / f"{comp_id}.json", 'w') as f:
        json.dump(comparison, f, indent=2)
    
    return comp_id

def load_all_jobs() -> List[Dict]:
    jobs = []
    for file in JOBS_DIR.glob("*.json"):
        try:
            with open(file, 'r') as f:
                jobs.append(json.load(f))
        except:
            pass
    return sorted(jobs, key=lambda x: x.get('date_added', ''), reverse=True)

def load_all_resumes() -> List[Dict]:
    resumes = []
    for file in RESUMES_DIR.glob("*.json"):
        try:
            with open(file, 'r') as f:
                resumes.append(json.load(f))
        except:
            pass
    return resumes

def load_all_comparisons() -> List[Dict]:
    comparisons = []
    for file in COMPARISONS_DIR.glob("*.json"):
        try:
            with open(file, 'r') as f:
                comparisons.append(json.load(f))
        except:
            pass
    return sorted(comparisons, key=lambda x: x.get('date', ''), reverse=True)

# UI Components
def render_sidebar():
    with st.sidebar:
        st.title("🎯 Job-Resume Matcher")
        
        pages = {
            "Dashboard": "📊",
            "New Comparison": "🔄",
            "Batch Processing": "🚀",
            "My Comparisons": "📁",
            "Job Library": "💼",
            "Resume Library": "📄",
            "Recommendations": "🎯",
            "Achievements": "🏆",
            "Settings": "⚙️",
            "Help/Support": "❓"
        }
        
        for page, icon in pages.items():
            if st.button(f"{icon} {page}", key=f"nav_{page}", use_container_width=True):
                st.session_state.current_page = page
                if page == "New Comparison":
                    st.session_state.wizard_stage = 1
        
        st.divider()
        
        # Quick stats
        st.caption("Quick Stats")
        comparisons = load_all_comparisons()
        jobs = load_all_jobs()
        resumes = load_all_resumes()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Comparisons", len(comparisons))
            st.metric("Jobs", len(jobs))
        with col2:
            st.metric("Resumes", len(resumes))
            avg_score = sum(c.get('match_score', 0) for c in comparisons) / len(comparisons) if comparisons else 0
            st.metric("Avg Score", f"{avg_score:.1f}%")
        
        st.divider()
        st.caption("Cache Performance")
        st.metric("Hit Rate", f"{cache.hit_rate*100:.1f}%")

def render_dashboard():
    st.header("📊 Dashboard")
    
    # Load data
    comparisons = load_all_comparisons()
    jobs = load_all_jobs()
    resumes = load_all_resumes()
    
    # Calculate user stats and achievements
    user_stats = gamification.calculate_user_stats(comparisons)
    
    # User Profile Section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.write("### 👤 Your Profile")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        level_title = gamification.get_level_title(user_stats['level'])
        st.markdown(f'<div class="match-score" style="font-size: 48px;">{level_title}</div>', unsafe_allow_html=True)
        st.write(f"**Level {user_stats['level']}** • {user_stats['xp']} / {user_stats['next_level_xp']} XP")
        
        # XP Progress bar
        progress = user_stats['xp'] / user_stats['next_level_xp'] if user_stats['next_level_xp'] > 0 else 0
        st.progress(progress)
        st.caption(f"Next level in {user_stats['next_level_xp'] - user_stats['xp']} XP")
    
    with col2:
        st.write("**📊 Your Stats**")
        st.write(f"• **Comparisons:** {user_stats['total_comparisons']}")
        st.write(f"• **Average Score:** {user_stats['average_score']:.1f}%")
        if user_stats['best_match']:
            st.write(f"• **Best Match:** {user_stats['best_match'].get('match_score', 0):.1f}%")
        if user_stats['improvement_rate'] > 0:
            st.write(f"• **Improvement:** +{user_stats['improvement_rate']:.1f}%")
    
    with col3:
        st.write("**🏆 Achievements**")
        if user_stats['achievements']:
            for achievement in user_stats['achievements'][:3]:  # Show top 3
                st.write(f"• {achievement['name']}")
            if len(user_stats['achievements']) > 3:
                st.caption(f"+{len(user_stats['achievements']) - 3} more")
        else:
            st.caption("No achievements yet")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Achievement Progress Section
    if user_stats['total_comparisons'] > 0:
        next_achievements = gamification.get_next_achievements(comparisons)
        if next_achievements:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.write("### 🎯 Next Achievements")
            
            for achievement in next_achievements:
                progress = achievement['progress'] / achievement['target']
                st.write(f"**{achievement['name']}** - {achievement['desc']}")
                st.progress(progress)
                st.caption(f"{achievement['progress']} / {achievement['target']}")
                st.divider()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Metrics with glassmorphism styling
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Comparisons", len(comparisons))
    with col2:
        st.metric("💼 Jobs in Library", len(jobs))
    with col3:
        st.metric("📄 Resumes in Library", len(resumes))
    with col4:
        avg_score = sum(c.get('match_score', 0) for c in comparisons) / len(comparisons) if comparisons else 0
        st.metric("🎯 Average Match Score", f"{avg_score:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Cache statistics with glassmorphism styling
    st.divider()
    st.subheader("⚡ Performance Metrics")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Cache Hit Rate", f"{cache.hit_rate*100:.1f}%")
    with col2:
        st.metric("✅ Cache Hits", cache.cache_stats["hits"])
    with col3:
        st.metric("❌ Cache Misses", cache.cache_stats["misses"])
    with col4:
        total_requests = cache.cache_stats["hits"] + cache.cache_stats["misses"]
        st.metric("📊 Total Requests", total_requests)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Analytics Insights
    if comparisons:
        st.subheader("📊 Analytics Insights")
        
        # Generate insights
        insights = analytics.generate_insights(comparisons)
        
        if insights:
            # Analytics metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                trend_delta = insights['trends']['change']
                st.metric(
                    "📈 Score Trend",
                    f"{insights['trends']['direction'].title()}",
                    delta=f"{trend_delta:+.1f}%",
                    delta_color="normal" if trend_delta > 0 else "inverse"
                )
            
            with col2:
                success_factors = insights['success_factors']
                success_rate = success_factors.get('success_rate', 0)
                st.metric(
                    "🎯 Success Rate",
                    f"{success_rate:.1f}%",
                    help="Percentage of comparisons with 80%+ match scores"
                )
            
            with col3:
                high_score_count = insights['success_factors'].get('high_score_count', 0)
                st.metric(
                    "🏆 High Scores",
                    high_score_count,
                    help="Number of comparisons with 80%+ match scores"
                )
            
            with col4:
                total_comparisons = len(comparisons)
                st.metric(
                    "📊 Total Comparisons",
                    total_comparisons,
                    help="Total number of comparisons made"
                )
            
            # Trend chart
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.plotly_chart(analytics.create_trend_chart(comparisons), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Skills analysis and recommendations
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.plotly_chart(analytics.create_skills_analysis_chart(comparisons), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.write("### 🎯 Top Skills in Successful Matches")
                skills_data = insights['top_skills']
                if skills_data['successful_skills']:
                    for skill, count in skills_data['successful_skills'][:5]:
                        st.write(f"• **{skill.title()}** - {count} times")
                else:
                    st.info("No successful matches yet to analyze skills")
                
                st.divider()
                
                st.write("### 💡 Recommendations")
                recommendations = insights['recommendations']
                if recommendations:
                    for rec in recommendations:
                        st.info(f"💡 {rec}")
                else:
                    st.success("🎉 Great job! Your comparisons are performing well.")
                st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Recent comparisons with glassmorphism styling
    st.subheader("📈 Recent Comparisons")
    if comparisons:
        recent = comparisons[:5]
        for comp in recent:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
            with col1:
                st.write(f"**{comp.get('job_title', 'Unknown Job')}**")
                st.caption(f"👤 {comp.get('candidate_name', 'Unknown Candidate')}")
            with col2:
                score = comp.get('match_score', 0)
                color = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
                st.write(f"{color} **{score}% Match**")
            with col3:
                st.write(f"📅 {comp.get('date', '')}")
            with col4:
                if st.button("👁️ View", key=f"view_{comp['id']}"):
                    st.session_state.current_page = "My Comparisons"
                    st.session_state.selected_comparison = comp['id']
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.info("📝 No comparisons yet. Start by creating a new comparison!")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Batch Processing Section
    with st.expander("🚀 Batch Processing", expanded=False):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("### 📊 Process Multiple Jobs and Resumes")
        st.info("Upload multiple job descriptions and resumes to compare all combinations at once.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            job_files = st.file_uploader(
                "Upload multiple job descriptions",
                type=['pdf', 'docx', 'txt'],
                accept_multiple_files=True,
                key="batch_jobs"
            )
            if job_files:
                st.success(f"✅ {len(job_files)} job files uploaded")
        
        with col2:
            resume_files = st.file_uploader(
                "Upload multiple resumes",
                type=['pdf', 'docx', 'txt'],
                accept_multiple_files=True,
                key="batch_resumes"
            )
            if resume_files:
                st.success(f"✅ {len(resume_files)} resume files uploaded")
        
        if job_files and resume_files:
            total_comparisons = len(job_files) * len(resume_files)
            st.warning(f"⚠️ This will create {total_comparisons} comparisons")
            
            if st.button("🎯 Start Batch Comparison", use_container_width=True):
                # Show skeleton loader during processing
                with st.container():
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.write("🔄 **Processing Batch Comparison...**")
                    show_skeleton_loader()
                    st.markdown('</div>', unsafe_allow_html=True)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress):
                    progress_bar.progress(progress)
                    status_text.text(f"Processing... {progress*100:.0f}% ({int(progress * total_comparisons)}/{total_comparisons})")
                
                try:
                    results_df = run_async(
                        batch_processor.batch_compare(
                            job_files, resume_files, update_progress
                        )
                    )
                    
                    st.success(f"✅ Completed {len(results_df)} comparisons!")
                    
                    # Create summary
                    summary = batch_processor.create_batch_summary(results_df)
                    
                    # Show summary metrics
                    if summary:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("📊 Total Comparisons", summary['total_comparisons'])
                        with col2:
                            st.metric("🎯 Average Score", f"{summary['average_score']:.1f}%")
                        with col3:
                            st.metric("🏆 Success Rate", f"{summary['success_rate']:.1f}%")
                        with col4:
                            st.metric("📈 Max Score", f"{summary['max_score']:.1f}%")
                    
                    # Show results table
                    st.write("### 📋 Comparison Results")
                    st.dataframe(
                        results_df[['job_title', 'candidate', 'match_score', 'overall_summary']].style.background_gradient(subset=['match_score']),
                        use_container_width=True
                    )
                    
                    # Show visualization
                    st.write("### 📊 Score Distribution")
                    st.plotly_chart(batch_processor.create_batch_visualization(results_df), use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results CSV",
                        csv,
                        f"batch_comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                    
                    # Show top performers
                    if summary:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("### 🏆 Top Performing Jobs")
                            for job, score in summary['top_jobs'].items():
                                st.write(f"• **{job}** - {score:.1f}% avg")
                        
                        with col2:
                            st.write("### 👑 Top Performing Candidates")
                            for candidate, score in summary['top_candidates'].items():
                                st.write(f"• **{candidate}** - {score:.1f}% avg")
                
                except Exception as e:
                    st.error(f"❌ Batch processing failed: {str(e)}")
                    st.info("Please check your files and try again.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Quick actions with glassmorphism styling
    st.subheader("⚡ Quick Actions")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 New Comparison", use_container_width=True):
            st.session_state.current_page = "New Comparison"
            st.session_state.wizard_stage = 1
    with col2:
        if st.button("💼 Add Job", use_container_width=True):
            st.session_state.current_page = "Job Library"
    with col3:
        if st.button("📄 Upload Resume", use_container_width=True):
            st.session_state.current_page = "Resume Library"
    st.markdown('</div>', unsafe_allow_html=True)

def render_new_comparison():
    st.header("🔄 New Comparison")
    
    # Progress indicator
    progress = st.session_state.wizard_stage / 3
    st.progress(progress)
    st.caption(f"Step {st.session_state.wizard_stage} of 3")
    
    if st.session_state.wizard_stage == 1:
        render_input_stage()
    elif st.session_state.wizard_stage == 2:
        render_review_stage()
    elif st.session_state.wizard_stage == 3:
        render_results_stage()

def render_batch_processing():
    st.header("🚀 Batch Processing")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.write("### 📊 Process Multiple Jobs and Resumes")
    st.info("Upload multiple job descriptions and resumes to compare all combinations at once. Perfect for recruiters and HR professionals!")
    st.markdown('</div>', unsafe_allow_html=True)

def render_achievements():
    st.header("🏆 Achievements & Progress")
    
    # Load data
    comparisons = load_all_comparisons()
    jobs = load_all_jobs()
    resumes = load_all_resumes()
    
    # Calculate user stats
    user_stats = gamification.calculate_user_stats(comparisons)
    
    # User Level Overview
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.write("### 👤 Your Progress")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        level_title = gamification.get_level_title(user_stats['level'])
        st.markdown(f'<div class="match-score" style="font-size: 36px;">{level_title}</div>', unsafe_allow_html=True)
        st.write(f"**Level {user_stats['level']}**")
    
    with col2:
        st.metric("🎯 Total XP", user_stats['xp'])
        st.metric("📈 Next Level", f"{user_stats['next_level_xp'] - user_stats['xp']} XP needed")
    
    with col3:
        progress = user_stats['xp'] / user_stats['next_level_xp'] if user_stats['next_level_xp'] > 0 else 0
        st.progress(progress)
        st.caption(f"Level Progress: {progress*100:.1f}%")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Achievements Section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.write("### 🏆 Your Achievements")
    
    if user_stats['achievements']:
        col1, col2 = st.columns(2)
        
        for i, achievement in enumerate(user_stats['achievements']):
            col = col1 if i % 2 == 0 else col2
            with col:
                st.success(f"✅ {achievement['name']}")
                st.caption(achievement['desc'])
                st.divider()
    else:
        st.info("🎯 No achievements yet! Complete comparisons to earn achievements.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Achievement Progress
    next_achievements = gamification.get_next_achievements(comparisons)
    if next_achievements:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("### 🎯 Next Achievements")
        st.info("Keep working towards these achievements!")
        
        for achievement in next_achievements:
            progress = achievement['progress'] / achievement['target']
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{achievement['name']}**")
                st.caption(achievement['desc'])
                st.progress(progress)
                st.caption(f"Progress: {achievement['progress']} / {achievement['target']}")
            
            with col2:
                st.metric("Progress", f"{progress*100:.0f}%")
            
            st.divider()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Statistics Section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.write("### 📊 Your Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Comparisons", user_stats['total_comparisons'])
    
    with col2:
        st.metric("🎯 Average Score", f"{user_stats['average_score']:.1f}%")
    
    with col3:
        if user_stats['best_match']:
            st.metric("🏆 Best Match", f"{user_stats['best_match'].get('match_score', 0):.1f}%")
        else:
            st.metric("🏆 Best Match", "N/A")
    
    with col4:
        if user_stats['improvement_rate'] > 0:
            st.metric("📈 Improvement", f"+{user_stats['improvement_rate']:.1f}%")
        else:
            st.metric("📈 Improvement", "0%")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # All Available Achievements
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.write("### 🎖️ All Available Achievements")
    
    earned_achievement_names = {a['name'] for a in user_stats['achievements']}
    
    col1, col2 = st.columns(2)
    achievement_list = list(gamification.achievements.items())
    
    for i, (key, achievement) in enumerate(achievement_list):
        col = col1 if i % 2 == 0 else col2
        
        with col:
            if achievement['name'] in earned_achievement_names:
                st.success(f"✅ {achievement['name']}")
            else:
                st.info(f"🔒 {achievement['name']}")
            
            st.caption(achievement['desc'])
            st.divider()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tips for earning achievements
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.write("### 💡 Tips to Earn More Achievements")
    
    tips = [
        "🎯 **Complete your first comparison** to earn the First Match badge",
        "⭐ **Aim for high scores** (80%+) to become a High Scorer",
        "💯 **Perfect your resume** to achieve 95%+ matches",
        "🎓 **Develop diverse skills** to match 20+ skills in one comparison",
        "📈 **Stay consistent** by doing 5 comparisons in a week",
        "🔍 **Explore different job types** to become an Explorer",
        "🚀 **Try batch processing** to earn the Batch Master badge",
        "📝 **Upload multiple resumes** to become a Resume Guru",
        "💼 **Add job descriptions** to your library",
        "📤 **Share your results** to become a Sharing Pro"
    ]
    
    for tip in tips:
        st.write(f"• {tip}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # File upload section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 💼 Job Descriptions")
        job_files = st.file_uploader(
            "Upload multiple job descriptions",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            key="batch_jobs_page"
        )
        if job_files:
            st.success(f"✅ {len(job_files)} job files uploaded")
            for file in job_files:
                st.caption(f"• {file.name} ({file.size / 1024:.1f} KB)")
    
    with col2:
        st.write("### 📄 Resumes")
        resume_files = st.file_uploader(
            "Upload multiple resumes",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            key="batch_resumes_page"
        )
        if resume_files:
            st.success(f"✅ {len(resume_files)} resume files uploaded")
            for file in resume_files:
                st.caption(f"• {file.name} ({file.size / 1024:.1f} KB)")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Processing section
    if job_files and resume_files:
        total_comparisons = len(job_files) * len(resume_files)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("### ⚙️ Processing Configuration")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Comparisons", total_comparisons)
        with col2:
            st.metric("💼 Job Files", len(job_files))
        with col3:
            st.metric("📄 Resume Files", len(resume_files))
        
        st.warning(f"⚠️ This will create {total_comparisons} comparisons. Processing time depends on file complexity.")
        
        if st.button("🎯 Start Batch Comparison", use_container_width=True, type="primary"):
            # Show skeleton loader during processing
            with st.container():
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.write("🔄 **Processing Batch Comparison...**")
                show_skeleton_loader()
                st.markdown('</div>', unsafe_allow_html=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(progress):
                progress_bar.progress(progress)
                status_text.text(f"Processing... {progress*100:.0f}% ({int(progress * total_comparisons)}/{total_comparisons})")
            
            try:
                results_df = run_async(
                    batch_processor.batch_compare(
                        job_files, resume_files, update_progress
                    )
                )
                
                st.success(f"✅ Completed {len(results_df)} comparisons!")
                
                # Create summary
                summary = batch_processor.create_batch_summary(results_df)
                
                # Show summary metrics
                if summary:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.write("### 📊 Batch Results Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📊 Total Comparisons", summary['total_comparisons'])
                    with col2:
                        st.metric("🎯 Average Score", f"{summary['average_score']:.1f}%")
                    with col3:
                        st.metric("🏆 Success Rate", f"{summary['success_rate']:.1f}%")
                    with col4:
                        st.metric("📈 Max Score", f"{summary['max_score']:.1f}%")
                    
                    # Score breakdown
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🟢 High Scores (80%+)", summary['high_scores'])
                    with col2:
                        st.metric("🟡 Medium Scores (60-79%)", summary['medium_scores'])
                    with col3:
                        st.metric("🔴 Low Scores (<60%)", summary['low_scores'])
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Show results table
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.write("### 📋 Detailed Results")
                st.dataframe(
                    results_df[['job_title', 'candidate', 'match_score', 'overall_summary']].style.background_gradient(subset=['match_score']),
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show visualization
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.write("### 📊 Score Distribution")
                st.plotly_chart(batch_processor.create_batch_visualization(results_df), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show top performers
                if summary:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("### 🏆 Top Performing Jobs")
                        for job, score in summary['top_jobs'].items():
                            st.write(f"• **{job}** - {score:.1f}% avg")
                    
                    with col2:
                        st.write("### 👑 Top Performing Candidates")
                        for candidate, score in summary['top_candidates'].items():
                            st.write(f"• **{candidate}** - {score:.1f}% avg")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Download section
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.write("### 📥 Download Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Full Results CSV",
                        csv,
                        f"batch_comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Create summary CSV
                    summary_df = pd.DataFrame([summary]) if summary else pd.DataFrame()
                    summary_csv = summary_df.to_csv(index=False)
                    st.download_button(
                        "📊 Download Summary CSV",
                        summary_csv,
                        f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"❌ Batch processing failed: {str(e)}")
                st.info("Please check your files and try again.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.info("📝 Please upload both job descriptions and resumes to start batch processing.")
        st.markdown('</div>', unsafe_allow_html=True)

def render_input_stage():
    st.subheader("Step 1: Input Job Description and Resume")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Job Description")
        input_method = st.radio("Input method:", ["Paste text", "Upload file", "Select from library"], key="job_input_method")
        
        job_text = ""
        if input_method == "Paste text":
            job_text = st.text_area("Paste job description:", height=300, key="job_paste")
        elif input_method == "Upload file":
            file = st.file_uploader("Upload job description", type=['pdf', 'docx', 'txt'], key="job_upload")
            if file:
                try:
                    job_text = extract_text_from_file(file)
                    st.success("File uploaded successfully!")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            jobs = load_all_jobs()
            if jobs:
                selected_job = st.selectbox("Select from library:", 
                    options=[(j['id'], f"{j['title']} - {j['company']}") for j in jobs],
                    format_func=lambda x: x[1],
                    key="job_select"
                )
                if selected_job:
                    job = next(j for j in jobs if j['id'] == selected_job[0])
                    job_text = job['raw_text']
            else:
                st.info("No jobs in library yet.")
    
    with col2:
        st.write("### Resume")
        input_method = st.radio("Input method:", ["Paste text", "Upload file", "Select from library"], key="resume_input_method")
        
        resume_text = ""
        if input_method == "Paste text":
            resume_text = st.text_area("Paste resume:", height=300, key="resume_paste")
        elif input_method == "Upload file":
            file = st.file_uploader("Upload resume", type=['pdf', 'docx', 'txt'], key="resume_upload")
            if file:
                try:
                    resume_text = extract_text_from_file(file)
                    st.success("File uploaded successfully!")
                    
                    # Save uploaded file
                    file_path = RESUME_FILES_DIR / f"{uuid.uuid4()}_{file.name}"
                    with open(file_path, 'wb') as f:
                        f.write(file.read())
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            resumes = load_all_resumes()
            if resumes:
                selected_resume = st.selectbox("Select from library:", 
                    options=[(r['id'], r.get('name', 'Unknown')) for r in resumes],
                    format_func=lambda x: x[1],
                    key="resume_select"
                )
                if selected_resume:
                    resume = next(r for r in resumes if r['id'] == selected_resume[0])
                    resume_text = resume['raw_text']
            else:
                st.info("No resumes in library yet.")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col3:
        if st.button("Next →", disabled=not (job_text and resume_text), use_container_width=True):
            # Create placeholders for real-time updates
            ai_thinking_placeholder = st.empty()
            progress_placeholder = st.empty()
            
            # Initialize thinking history
            job_thoughts = []
            resume_thoughts = []
            
            # Initialize progress in session state if not exists
            if 'job_progress' not in st.session_state:
                st.session_state.job_progress = StreamingProgress()
            if 'resume_progress' not in st.session_state:
                st.session_state.resume_progress = StreamingProgress()
            
            # Callback functions
            def job_thought_callback(thought: AIThought):
                job_thoughts.append(thought)
                if 'job_progress' in st.session_state:
                    render_ai_thinking_panel(ai_thinking_placeholder, st.session_state.job_progress, job_thoughts)
            
            def resume_thought_callback(thought: AIThought):
                resume_thoughts.append(thought)
                if 'resume_progress' in st.session_state:
                    render_ai_thinking_panel(ai_thinking_placeholder, st.session_state.resume_progress, resume_thoughts)
            
            def job_progress_callback(progress: StreamingProgress):
                st.session_state.job_progress = progress
                with progress_placeholder.container():
                    st.write("### 💼 Processing Job Description")
                    render_ai_thinking_panel(st.empty(), progress, job_thoughts)
            
            def resume_progress_callback(progress: StreamingProgress):
                st.session_state.resume_progress = progress
                with progress_placeholder.container():
                    st.write("### 📄 Processing Resume")
                    render_ai_thinking_panel(st.empty(), progress, resume_thoughts)
            
            # Check model availability
            connected, models = check_ollama_connection()
            
            if connected:
                # Process job description with streaming
                st.write("### 💼 Processing Job Description")
                job_result = None
                job_confidence = 0.0
                
                for partial_result, progress in llm_parse_streaming(
                    job_text, 'job', 
                    progress_callback=job_progress_callback,
                    thought_callback=job_thought_callback
                ):
                    job_result = partial_result
                    job_confidence = progress.overall_confidence
                
                st.session_state.parsed_job = job_result
                st.session_state.job_confidence = job_confidence
                
                # Process resume with streaming
                st.write("### 📄 Processing Resume")
                resume_result = None
                resume_confidence = 0.0
                
                for partial_result, progress in llm_parse_streaming(
                    resume_text, 'resume',
                    progress_callback=resume_progress_callback,
                    thought_callback=resume_thought_callback
                ):
                    resume_result = partial_result
                    resume_confidence = progress.overall_confidence
                
                st.session_state.parsed_resume = resume_result
                st.session_state.resume_confidence = resume_confidence
                
                # Show final summary
                st.success(f"✅ Job parsed with {job_confidence*100:.1f}% confidence")
                st.success(f"✅ Resume parsed with {resume_confidence*100:.1f}% confidence")
                
                # Display AI insights
                with st.expander("🧠 AI Processing Insights", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Job Processing Summary:**")
                        st.write(f"- Total tokens: {sum(t.tokens_generated for t in job_thoughts)}")
                        st.write(f"- Processing stages: {len(set(t.stage for t in job_thoughts))}")
                        st.write(f"- Average confidence: {job_confidence*100:.1f}%")
                        
                        if job_result.get('quality_score'):
                            st.write(f"- Document quality: {job_result['quality_score']}")
                    
                    with col2:
                        st.write("**Resume Processing Summary:**")
                        st.write(f"- Total tokens: {sum(t.tokens_generated for t in resume_thoughts)}")
                        st.write(f"- Processing stages: {len(set(t.stage for t in resume_thoughts))}")
                        st.write(f"- Average confidence: {resume_confidence*100:.1f}%")
                        
                        if resume_result.get('quality_score'):
                            st.write(f"- Document quality: {resume_result['quality_score']}")
            else:
                st.error("❌ AI model not available. Please install transformers and torch.")
                st.stop()
            
            # Add raw text
            st.session_state.parsed_job['raw_text'] = job_text
            st.session_state.parsed_resume['raw_text'] = resume_text
            
            # Store AI insights for later use
            st.session_state.job_ai_insights = {
                'thoughts': job_thoughts,
                'confidence': job_confidence,
                'token_count': sum(t.tokens_generated for t in job_thoughts)
            }
            st.session_state.resume_ai_insights = {
                'thoughts': resume_thoughts,
                'confidence': resume_confidence,
                'token_count': sum(t.tokens_generated for t in resume_thoughts)
            }
            
            st.session_state.wizard_stage = 2
            st.rerun()

def render_review_stage():
    st.subheader("Step 2: Review & Edit Parsed Data")
    
    # Show confidence scores if available
    if hasattr(st.session_state, 'job_confidence'):
        col1, col2 = st.columns(2)
        with col1:
            job_conf = st.session_state.job_confidence
            # Use emoji indicators instead of colors
            conf_indicator = "🟢" if job_conf >= 0.7 else "🟡" if job_conf >= 0.5 else "🔴"
            st.metric(
                "Job Parsing Confidence", 
                f"{conf_indicator} {job_conf*100:.1f}%",
                help="🟢 High (70%+) | 🟡 Medium (50-70%) | 🔴 Low (<50%)"
            )
            
        with col2:
            resume_conf = st.session_state.resume_confidence
            conf_indicator = "🟢" if resume_conf >= 0.7 else "🟡" if resume_conf >= 0.5 else "🔴"
            st.metric(
                "Resume Parsing Confidence", 
                f"{conf_indicator} {resume_conf*100:.1f}%",
                help="🟢 High (70%+) | 🟡 Medium (50-70%) | 🔴 Low (<50%)"
            )
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 💼 Job Description")
        job = st.session_state.parsed_job
        
        job['title'] = st.text_input("**Job Title:**", value=job.get('title', ''), key="job_title_input")
        job['company'] = st.text_input("**Company:**", value=job.get('company', ''), key="job_company_input")
        job['location'] = st.text_input("**Location:**", value=job.get('location', ''), key="job_location_input")
        job['type'] = st.selectbox("**Employment Type:**", 
            options=["Full-time", "Part-time", "Contract", "Remote", "Hybrid", "Freelance"],
            index=["Full-time", "Part-time", "Contract", "Remote", "Hybrid", "Freelance"].index(
                job.get('type', 'Full-time')
            ) if job.get('type', 'Full-time') in ["Full-time", "Part-time", "Contract", "Remote", "Hybrid", "Freelance"] else 0,
            key="job_type_select"
        )
        
        st.write("**Required Skills:**")
        current_skills = job.get('required_skills', [])
        skills_text = st.text_area(
            "Enter skills (one per line):", 
            value='\n'.join(current_skills) if current_skills else '',
            height=150,
            help="Add each skill on a new line",
            key="job_skills_textarea"  # ADD THIS KEY
        )
        job['required_skills'] = [s.strip() for s in skills_text.split('\n') if s.strip()]
        
        job['experience'] = st.text_input("**Experience Required:**", 
            value=job.get('experience', ''),
            help="e.g., 5+ years, 3-5 years",
            key="job_experience_input")
        job['education'] = st.text_input("**Education Required:**", 
            value=job.get('education', ''),
            help="e.g., Bachelor's degree in Computer Science",
            key="job_education_input")
        
        # Show parsed count
        st.caption(f"📊 Found {len(job.get('required_skills', []))} required skills")
    
    with col2:
        st.write("### 📄 Resume")
        resume = st.session_state.parsed_resume
        
        resume['name'] = st.text_input("**Name:**", value=resume.get('name', ''), key="resume_name_input")
        resume['email'] = st.text_input("**Email:**", value=resume.get('email', ''), key="resume_email_input")
        resume['phone'] = st.text_input("**Phone:**", value=resume.get('phone', ''), key="resume_phone_input")
        
        st.write("**Skills:**")
        current_skills = resume.get('skills', [])
        skills_text = st.text_area(
            "Enter skills (one per line):", 
            value='\n'.join(current_skills) if current_skills else '',
            height=150,
            help="Add each skill on a new line",
            key="resume_skills_textarea"  # ADD THIS KEY
        )
        resume['skills'] = [s.strip() for s in skills_text.split('\n') if s.strip()]
        
        resume['education'] = st.text_input("**Education:**", 
            value=resume.get('education', ''),
            help="e.g., BS Computer Science from MIT",
            key="resume_education_input")
        
        # Experience display
        st.write("**Experience Summary:**")
        exp_text = ""
        for exp in resume.get('experience', []):
            title = exp.get('title', 'No title')
            company = exp.get('company', 'No company')
            duration = exp.get('duration', 'No duration')
            exp_text += f"• {title} at {company} ({duration})\n"
        
        if exp_text:
            st.text_area("Experience:", value=exp_text, height=100, disabled=True, key="resume_experience_display")
        else:
            st.info("No experience data extracted. You may want to add it manually.")
        
        # Show parsed count
        st.caption(f"📊 Found {len(resume.get('skills', []))} skills")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.wizard_stage = 1
            st.rerun()
    with col3:
        if st.button("Analyze →", use_container_width=True):
            # Show skeleton loader during analysis
            with st.container():
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.write("🧠 **Analyzing Match with AI...**")
                show_skeleton_loader()
                st.markdown('</div>', unsafe_allow_html=True)
            
            with st.spinner("Analyzing match..."):
                score, results = calculate_match_score(
                    st.session_state.parsed_job,
                    st.session_state.parsed_resume
                )
                st.session_state.comparison_results = results
                st.session_state.match_score = score
                st.session_state.wizard_stage = 3
                st.rerun()

def render_results_stage():
    st.subheader("Step 3: Comparison Results")
    
    results = st.session_state.comparison_results
    score = st.session_state.match_score
    job = st.session_state.parsed_job
    resume = st.session_state.parsed_resume
    
    # Score display with animated styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="match-score">{score:.1f}%</div>', unsafe_allow_html=True)
        st.write(f"**{results.overall_summary}**")
        
        # Visual score indicator
        color = "green" if score >= 80 else "orange" if score >= 60 else "red"
        st.progress(score / 100)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Visualizations
    st.write("### 📊 Match Analysis Visualizations")
    
    # Radar chart
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_match_visualization(results), use_container_width=True)
    
    # Skills gap chart
    with col2:
        st.plotly_chart(create_skill_gap_chart(results.skill_matches_detailed), use_container_width=True)
    
    # Skills sunburst chart
    st.plotly_chart(create_skill_sunburst(results.skill_matches_detailed), use_container_width=True)
    
    st.divider()
    
    # ====================================================================================
    # NEW SECTION: DYNAMIC INTERVIEW QUESTIONS
    # ====================================================================================
    
    st.write("### 🎤 AI-Generated Interview Questions")
    
    # Initialize question generator
    question_generator = DynamicQuestionGenerator()
    
    # Generate questions with loading animation
    with st.spinner("🧠 AI is crafting role-specific interview questions..."):
        interview_questions = question_generator.generate_interview_questions(
            job, resume, results, num_questions=12
        )
    
    # Display questions in organized tabs
    question_tabs = st.tabs([
        "Technical", "Behavioral", "Situational", 
        "Experience", "Cultural Fit", "Growth", "Follow-ups"
    ])
    
    tab_mapping = {
        0: 'technical',
        1: 'behavioral', 
        2: 'situational',
        3: 'experience',
        4: 'cultural',
        5: 'growth',
        6: 'follow_ups'
    }
    
    for idx, tab in enumerate(question_tabs):
        with tab:
            category = tab_mapping.get(idx, 'technical')
            questions = interview_questions.get(category, [])
            
            if questions:
                for i, q in enumerate(questions, 1):
                    with st.expander(f"Question {i}: {q['question'][:50]}...", expanded=i==1):
                        st.write(f"**Full Question:** {q['question']}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Why This Matters:**")
                            st.caption(q.get('importance', 'Evaluates key competencies'))
                            
                            st.write("**What to Look For:**")
                            st.caption(q.get('look_for', 'Clear examples and structured thinking'))
                        
                        with col2:
                            st.write("**Red Flags:**")
                            st.caption(q.get('red_flags', 'Vague or evasive answers'))
                            
                            difficulty_color = {
                                'basic': '🟢',
                                'intermediate': '🟡',
                                'advanced': '🔴'
                            }
                            st.write(f"**Difficulty:** {difficulty_color.get(q.get('difficulty', 'intermediate'), '🟡')} {q.get('difficulty', 'intermediate').title()}")
                            st.write(f"**Time Estimate:** {q.get('time_estimate', '2-3 minutes')}")
            else:
                st.info("No questions generated for this category")
    
    # Download questions
    all_questions = []
    for category, questions in interview_questions.items():
        for q in questions:
            all_questions.append({
                'Category': category,
                'Question': q['question'],
                'Importance': q.get('importance', ''),
                'Look For': q.get('look_for', ''),
                'Red Flags': q.get('red_flags', ''),
                'Difficulty': q.get('difficulty', ''),
                'Time': q.get('time_estimate', '')
            })
    
    if all_questions:
        questions_df = pd.DataFrame(all_questions)
        csv = questions_df.to_csv(index=False)
        st.download_button(
            "📥 Download All Interview Questions",
            csv,
            f"interview_questions_{job.get('title', 'role').replace(' ', '_')}.csv",
            "text/csv"
        )
    
    st.divider()
    
    # ====================================================================================
    # NEW SECTION: CONTEXTUAL TIPS
    # ====================================================================================
    
    st.write("### 💡 Personalized Action Tips")
    
    # Initialize tip generator
    tip_generator = ContextualTipGenerator()
    
    # Generate tips with loading animation
    with st.spinner("🤖 AI is analyzing your profile for personalized recommendations..."):
        contextual_tips = tip_generator.generate_contextual_tips(job, resume, results)
    
    # Display priority actions first
    if 'priority_actions' in contextual_tips:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("#### 🎯 Priority Actions")
        
        for action in contextual_tips['priority_actions']:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                priority_emoji = {
                    'critical': '🔴',
                    'high': '🟡',
                    'medium': '🟢'
                }
                st.write(f"{priority_emoji.get(action['priority'], '🟢')} **{action['action']}**")
                st.caption(f"Reason: {action['reason']}")
            with col2:
                st.metric("Deadline", action['deadline'])
            with col3:
                st.metric("Impact", action['impact'].title())
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display categorized tips
    tip_tabs = st.tabs([
        "Resume", "Skills", "Interview", "Networking", "Application", "Branding"
    ])
    
    tip_tab_mapping = {
        0: 'resume_improvement',
        1: 'skill_development',
        2: 'interview_preparation',
        3: 'networking',
        4: 'application_strategy',
        5: 'personal_branding'
    }
    
    for idx, tab in enumerate(tip_tabs):
        with tab:
            category = tip_tab_mapping.get(idx)
            tips = contextual_tips.get(category, [])
            
            if tips:
                for i, tip in enumerate(tips, 1):
                    with st.expander(f"Tip {i}: {tip['tip'][:50]}...", expanded=i==1):
                        st.write(f"**Action:** {tip['tip']}")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            impact_color = {
                                'high': '🔴',
                                'medium': '🟡',
                                'low': '🟢'
                            }
                            st.metric("Impact", f"{impact_color.get(tip['impact'], '🟡')} {tip['impact'].title()}")
                        
                        with col2:
                            effort_color = {
                                'low': '🟢',
                                'medium': '🟡',
                                'high': '🔴'
                            }
                            st.metric("Effort", f"{effort_color.get(tip['effort'], '🟡')} {tip['effort'].title()}")
                        
                        with col3:
                            st.metric("Timeframe", tip['timeframe'])
                        
                        with col4:
                            st.metric("Priority", tip['priority'].title())
                        
                        if tip.get('resources'):
                            st.write("**Resources:**")
                            for resource in tip['resources']:
                                st.write(f"• {resource}")
            else:
                st.info("No specific tips for this category")
    
    # Export all tips
    all_tips = []
    for category, tips in contextual_tips.items():
        if category != 'priority_actions':
            for tip in tips:
                all_tips.append({
                    'Category': tip_generator.tip_categories.get(category, category),
                    'Tip': tip['tip'],
                    'Priority': tip['priority'],
                    'Impact': tip['impact'],
                    'Effort': tip['effort'],
                    'Timeframe': tip['timeframe']
                })
    
    if all_tips:
        tips_df = pd.DataFrame(all_tips)
        csv = tips_df.to_csv(index=False)
        st.download_button(
            "📥 Download All Tips",
            csv,
            f"personalized_tips_{job.get('title', 'role').replace(' ', '_')}.csv",
            "text/csv"
        )
    
    st.divider()
    
    # Detailed results (existing code continues...)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("### 🎯 Skills Analysis")
        skill_details = results.skill_matches_detailed
        
        st.metric("🎯 Skills Match", f"{skill_details['score']:.1f}%")
        
        if skill_details['matched']:
            st.write("✅ **Matched Skills:**")
            for skill in skill_details['matched'][:10]:
                st.write(f"• {skill}")
        
        if skill_details['missing']:
            st.write("❌ **Missing Skills:**")
            for skill in skill_details['missing'][:10]:
                st.write(f"• {skill}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("### 💼 Experience & Education")
        
        st.metric("💼 Experience", results.experience_match_status)
        st.metric("🎓 Education", results.education_match_status)
        
        st.write("### 🤖 AI Recommendations")
        for rec in results.ai_recommendations:
            st.info(f"💡 {rec}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Smart Recommendations (keep existing code)
    st.write("### 🎯 Personalized Action Plan")
    
    # Generate smart recommendations
    smart_recs = rec_engine.generate_smart_recommendations(job, resume, results)
    
    # Success probability and timeline with glassmorphism styling
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Success Probability", f"{smart_recs['success_probability']:.1f}%")
    with col2:
        st.metric("⏱️ Timeline", smart_recs['estimated_timeline'])
    with col3:
        st.metric("📚 Skills to Learn", len(smart_recs['skill_roadmap']))
    st.markdown('</div>', unsafe_allow_html=True)

def render_comparisons():
    st.header("📁 My Comparisons")
    
    comparisons = load_all_comparisons()
    
    if not comparisons:
        st.info("No comparisons yet. Create your first comparison!")
        if st.button("Create New Comparison"):
            st.session_state.current_page = "New Comparison"
            st.rerun()
        return
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        score_filter = st.selectbox("Filter by score:", 
            ["All", "Excellent (80%+)", "Good (60-79%)", "Needs Work (<60%)"])
    with col2:
        sort_by = st.selectbox("Sort by:", ["Date", "Score", "Job Title", "Candidate"])
    with col3:
        order = st.selectbox("Order:", ["Descending", "Ascending"])
    
    # Apply filters
    filtered = comparisons
    if score_filter != "All":
        if "Excellent" in score_filter:
            filtered = [c for c in filtered if c.get('match_score', 0) >= 80]
        elif "Good" in score_filter:
            filtered = [c for c in filtered if 60 <= c.get('match_score', 0) < 80]
        else:
            filtered = [c for c in filtered if c.get('match_score', 0) < 60]
    
    # Sort
    reverse = order == "Descending"
    if sort_by == "Score":
        filtered.sort(key=lambda x: x.get('match_score', 0), reverse=reverse)
    elif sort_by == "Job Title":
        filtered.sort(key=lambda x: x.get('job_title', ''), reverse=reverse)
    elif sort_by == "Candidate":
        filtered.sort(key=lambda x: x.get('candidate_name', ''), reverse=reverse)
    
    # Display
    for comp in filtered:
        with st.expander(f"{comp.get('job_title', 'Unknown')} - {comp.get('candidate_name', 'Unknown')} ({comp.get('match_score', 0):.1f}%)"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**Date:** {comp.get('date', '')}")
                st.write(f"**Score:** {comp.get('match_score', 0):.1f}%")
                st.write(f"**Status:** {comp.get('status', 'Unknown')}")
                
                if comp.get('user_notes'):
                    st.write(f"**Notes:** {comp.get('user_notes')}")
                
                # Show key results
                results = comp.get('comparison_results', {})
                if results:
                    st.write("**Key Findings:**")
                    st.write(f"• {results.get('overall_summary', '')}")
                    st.write(f"• Skills Match: {results.get('skill_matches_detailed', {}).get('score', 0):.1f}%")
                    st.write(f"• {results.get('experience_match_status', '')}")
            
            with col2:
                if st.button("View Details", key=f"view_detail_{comp['id']}"):
                    st.session_state.selected_comparison = comp
                    st.session_state.show_comparison_detail = True
                    st.rerun()
                
                if st.button("Delete", key=f"delete_{comp['id']}"):
                    os.remove(COMPARISONS_DIR / f"{comp['id']}.json")
                    st.success("Comparison deleted!")
                    st.rerun()
    
    # Show comparison detail if selected
    if hasattr(st.session_state, 'show_comparison_detail') and st.session_state.show_comparison_detail:
        if hasattr(st.session_state, 'selected_comparison') and st.session_state.selected_comparison:
            render_comparison_detail(st.session_state.selected_comparison)

def render_comparison_detail(comparison: Dict):
    """Render detailed view of a comparison with visualizations"""
    st.header("📊 Comparison Details")
    
    # Back button
    if st.button("← Back to Comparisons"):
        st.session_state.show_comparison_detail = False
        st.session_state.selected_comparison = None
        st.rerun()
    
    st.divider()
    
    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Job Title", comparison.get('job_title', 'Unknown'))
    with col2:
        st.metric("Candidate", comparison.get('candidate_name', 'Unknown'))
    with col3:
        score = comparison.get('match_score', 0)
        st.metric("Match Score", f"{score:.1f}%")
    with col4:
        st.metric("Date", comparison.get('date', 'Unknown'))
    
    st.divider()
    
    # Visualizations
    st.write("### 📈 Match Analysis Visualizations")
    
    comparison_results = comparison.get('comparison_results', {})
    if comparison_results:
        # Convert dict back to ComparisonResult object for visualization
        results_obj = ComparisonResult(
            match_score=comparison_results.get('match_score', 0),
            overall_summary=comparison_results.get('overall_summary', ''),
            skill_matches_detailed=comparison_results.get('skill_matches_detailed', {}),
            experience_match_status=comparison_results.get('experience_match_status', ''),
            education_match_status=comparison_results.get('education_match_status', ''),
            ai_recommendations=comparison_results.get('ai_recommendations', []),
            missing_keywords=comparison_results.get('missing_keywords', []),
            ai_confidence_scores=comparison_results.get('ai_confidence_scores', {})
        )
        
        # Radar chart
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_match_visualization(results_obj), use_container_width=True)
        
        # Skills gap chart
        with col2:
            st.plotly_chart(create_skill_gap_chart(results_obj.skill_matches_detailed), use_container_width=True)
        
        # Skills sunburst chart
        st.plotly_chart(create_skill_sunburst(results_obj.skill_matches_detailed), use_container_width=True)
    
    st.divider()
    
    # Detailed breakdown
    st.write("### 📋 Detailed Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Skills Analysis")
        if comparison_results:
            skill_details = comparison_results.get('skill_matches_detailed', {})
            st.metric("Skills Match", f"{skill_details.get('score', 0):.1f}%")
            
            if skill_details.get('matched'):
                st.write("✅ **Matched Skills:**")
                for skill in skill_details['matched'][:15]:
                    st.write(f"• {skill}")
            
            if skill_details.get('missing'):
                st.write("❌ **Missing Skills:**")
                for skill in skill_details['missing'][:15]:
                    st.write(f"• {skill}")
    
    with col2:
        st.write("#### Experience & Education")
        if comparison_results:
            st.metric("Experience", comparison_results.get('experience_match_status', ''))
            st.metric("Education", comparison_results.get('education_match_status', ''))
            
            st.write("#### AI Recommendations")
            for rec in comparison_results.get('ai_recommendations', []):
                st.info(f"💡 {rec}")
    
    # Smart Recommendations for saved comparisons
    if comparison.get('parsed_data'):
        st.divider()
        st.write("### 🎯 Smart Recommendations")
        
        # Generate recommendations using saved data
        job_data = comparison['parsed_data'].get('job', {})
        resume_data = comparison['parsed_data'].get('resume', {})
        
        # Create ComparisonResult object from saved data
        saved_results = ComparisonResult(
            match_score=comparison.get('match_score', 0),
            overall_summary=comparison_results.get('overall_summary', ''),
            skill_matches_detailed=comparison_results.get('skill_matches_detailed', {}),
            experience_match_status=comparison_results.get('experience_match_status', ''),
            education_match_status=comparison_results.get('education_match_status', ''),
            ai_recommendations=comparison_results.get('ai_recommendations', []),
            missing_keywords=comparison_results.get('missing_keywords', []),
            ai_confidence_scores=comparison_results.get('ai_confidence_scores', {})
        )
        
        smart_recs = rec_engine.generate_smart_recommendations(job_data, resume_data, saved_results)
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Success Probability", f"{smart_recs['success_probability']:.1f}%")
        with col2:
            st.metric("Timeline", smart_recs['estimated_timeline'])
        with col3:
            st.metric("Skills to Learn", len(smart_recs['skill_roadmap']))
        
        # Show top recommendations
        if smart_recs['skill_roadmap']:
            st.write("**Top Skills to Develop:**")
            for i, skill in enumerate(smart_recs['skill_roadmap'][:3]):
                st.write(f"{i+1}. **{skill['skill']}** - {skill['time_estimate']} ({skill['difficulty']})")
        
        if smart_recs['resume_optimizations']:
            st.write("**Resume Improvements:**")
            for improvement in smart_recs['resume_optimizations'][:2]:
                st.info(f"💡 {improvement['suggestion']}")
    
    # Raw data
    if comparison.get('parsed_data'):
        st.divider()
        st.write("### 📄 Raw Data")
        
        parsed_data = comparison['parsed_data']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Job Data")
            job_data = parsed_data.get('job', {})
            st.json(job_data)
        
        with col2:
            st.write("#### Resume Data")
            resume_data = parsed_data.get('resume', {})
            st.json(resume_data)
    
    # User notes
    if comparison.get('user_notes'):
        st.divider()
        st.write("### 📝 User Notes")
        st.write(comparison['user_notes'])
    
    # Sharing options for saved comparisons
    st.divider()
    st.write("### 📤 Share This Comparison")
    
    share_col1, share_col2, share_col3 = st.columns(3)
    
    with share_col1:
        if st.button("🔗 Create Share Link", key=f"share_{comparison['id']}", use_container_width=True):
            url, token = sharing.create_shareable_link(comparison['id'])
            st.success("Share link created!")
            st.code(url)
            
            # Show QR code
            qr_img = sharing.generate_qr_code(url)
            st.image(f"data:image/png;base64,{qr_img}", width=200)
    
    with share_col2:
        if st.button("📄 Generate Report", key=f"report_{comparison['id']}", use_container_width=True):
            pdf_data = sharing.create_professional_report(comparison)
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_data,
                file_name=f"match_report_{comparison['id']}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    
    with share_col3:
        st.write("**Share Features:**")
        st.write("• Secure 48-hour links")
        st.write("• Mobile-friendly QR codes")
        st.write("• Professional PDF reports")
        st.write("• View tracking available")

def render_recommendations():
    st.header("🎯 Smart Recommendations")
    
    # Get all comparisons
    comparisons = load_all_comparisons()
    
    if not comparisons:
        st.info("No comparisons yet. Create your first comparison to get personalized recommendations!")
        if st.button("Create New Comparison"):
            st.session_state.current_page = "New Comparison"
            st.rerun()
        return
    
    # Select comparison for detailed recommendations
    st.write("### Select a comparison for detailed recommendations:")
    
    comparison_options = [(c['id'], f"{c.get('job_title', 'Unknown')} - {c.get('candidate_name', 'Unknown')} ({c.get('match_score', 0):.1f}%)") for c in comparisons]
    
    selected_comparison = st.selectbox(
        "Choose a comparison:",
        options=comparison_options,
        format_func=lambda x: x[1],
        key="recommendation_select"
    )
    
    if selected_comparison:
        comparison = next(c for c in comparisons if c['id'] == selected_comparison[0])
        
        # Generate comprehensive recommendations
        if comparison.get('parsed_data'):
            job_data = comparison['parsed_data'].get('job', {})
            resume_data = comparison['parsed_data'].get('resume', {})
            comparison_results = comparison.get('comparison_results', {})
            
            # Create ComparisonResult object
            results_obj = ComparisonResult(
                match_score=comparison.get('match_score', 0),
                overall_summary=comparison_results.get('overall_summary', ''),
                skill_matches_detailed=comparison_results.get('skill_matches_detailed', {}),
                experience_match_status=comparison_results.get('experience_match_status', ''),
                education_match_status=comparison_results.get('education_match_status', ''),
                ai_recommendations=comparison_results.get('ai_recommendations', []),
                missing_keywords=comparison_results.get('missing_keywords', []),
                ai_confidence_scores=comparison_results.get('ai_confidence_scores', {})
            )
            
            smart_recs = rec_engine.generate_smart_recommendations(job_data, resume_data, results_obj)
            
            # Display comprehensive recommendations
            st.divider()
            st.write("### 📊 Analysis Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Match", f"{comparison.get('match_score', 0):.1f}%")
            with col2:
                st.metric("Success Probability", f"{smart_recs['success_probability']:.1f}%")
            with col3:
                st.metric("Timeline", smart_recs['estimated_timeline'])
            with col4:
                st.metric("Skills to Learn", len(smart_recs['skill_roadmap']))
            
            # Detailed recommendations in tabs
            tabs = st.tabs(["📚 Skill Roadmap", "📝 Resume Tips", "⚡ Quick Wins", "📈 Progress Tracking"])
            
            with tabs[0]:
                if smart_recs['skill_roadmap']:
                    st.write("### Detailed Skill Development Plan")
                    for i, skill in enumerate(smart_recs['skill_roadmap']):
                        with st.container():
                            st.write(f"**{i+1}. {skill['skill'].title()}**")
                            
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                if skill['foundation']:
                                    st.caption(f"🔗 Build on: {', '.join(skill['foundation'])}")
                                else:
                                    st.caption("🆕 New skill area")
                                
                                # Progress indicator based on difficulty
                                progress = DIFFICULTY_PROGRESS.get(
                                    skill.get('difficulty'), DEFAULT_PROGRESS
                                )
                                st.progress(progress)
                            
                            with col2:
                                st.metric("Time", skill['time_estimate'])
                            
                            with col3:
                                st.metric("Difficulty", skill['difficulty'])
                            
                            # Learning resources
                            with st.expander("📖 Learning Resources"):
                                for resource in skill['resources']:
                                    st.write(f"• {resource}")
                            
                            st.divider()
                else:
                    st.success("🎉 Great job! No major skill gaps identified.")
            
            with tabs[1]:
                if smart_recs['resume_optimizations']:
                    st.write("### Resume Optimization Suggestions")
                    for improvement in smart_recs['resume_optimizations']:
                        priority_color = "red" if improvement['priority'] == 'High' else "orange"
                        st.write(f"**{improvement['priority']} Priority**")
                        st.info(f"💡 {improvement['suggestion']}")
                        st.caption(f"Example: {improvement['example']}")
                        st.divider()
                else:
                    st.success("✅ Your resume looks well-optimized for this position!")
            
            with tabs[2]:
                st.write("### Immediate Actions You Can Take:")
                for action in smart_recs['immediate_actions']:
                    st.write(f"• {action}")
                
                st.divider()
                st.write("### Next Steps:")
                st.write("1. Focus on the top 2-3 skills from the roadmap")
                st.write("2. Update your resume with the suggested improvements")
                st.write("3. Practice interview questions specific to this role")
                st.write("4. Research the company culture and values")
                st.write("5. Set up a study schedule for skill development")
            
            with tabs[3]:
                st.write("### Progress Tracking")
                
                # Create a simple progress tracker
                st.write("**Track your skill development progress:**")
                
                for skill in smart_recs['skill_roadmap'][:5]:  # Top 5 skills
                    progress = st.slider(
                        f"Progress on {skill['skill']}:",
                        0, 100, 0,
                        help=f"Track your progress learning {skill['skill']}"
                    )
                    
                    if progress > 0:
                        st.progress(progress / 100)
                        if progress >= 100:
                            st.success(f"🎉 Completed {skill['skill']}!")
                        elif progress >= 75:
                            st.info(f"Almost there! {100 - progress}% remaining")
                        elif progress >= 50:
                            st.warning(f"Halfway through {skill['skill']}")
                
                st.divider()
                st.write("**Study Schedule Suggestions:**")
                st.write("• Dedicate 1-2 hours daily to skill development")
                st.write("• Focus on one skill at a time")
                st.write("• Practice with real projects")
                st.write("• Review progress weekly")

def render_shared_comparison(comparison: Dict):
    """Render shared comparison view (read-only)"""
    
    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Job Title", comparison.get('job_title', 'Unknown'))
    with col2:
        st.metric("Candidate", comparison.get('candidate_name', 'Unknown'))
    with col3:
        score = comparison.get('match_score', 0)
        st.metric("Match Score", f"{score:.1f}%")
    with col4:
        st.metric("Date", comparison.get('date', 'Unknown'))
    
    st.divider()
    
    # Visualizations
    st.write("### 📈 Match Analysis Visualizations")
    
    comparison_results = comparison.get('comparison_results', {})
    if comparison_results:
        # Convert dict back to ComparisonResult object for visualization
        results_obj = ComparisonResult(
            match_score=comparison.get('match_score', 0),
            overall_summary=comparison_results.get('overall_summary', ''),
            skill_matches_detailed=comparison_results.get('skill_matches_detailed', {}),
            experience_match_status=comparison_results.get('experience_match_status', ''),
            education_match_status=comparison_results.get('education_match_status', ''),
            ai_recommendations=comparison_results.get('ai_recommendations', []),
            missing_keywords=comparison_results.get('missing_keywords', []),
            ai_confidence_scores=comparison_results.get('ai_confidence_scores', {})
        )
        
        # Radar chart
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_match_visualization(results_obj), use_container_width=True)
        
        # Skills gap chart
        with col2:
            st.plotly_chart(create_skill_gap_chart(results_obj.skill_matches_detailed), use_container_width=True)
        
        # Skills sunburst chart
        st.plotly_chart(create_skill_sunburst(results_obj.skill_matches_detailed), use_container_width=True)
    
    st.divider()
    
    # Detailed breakdown
    st.write("### 📋 Detailed Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Skills Analysis")
        if comparison_results:
            skill_details = comparison_results.get('skill_matches_detailed', {})
            st.metric("Skills Match", f"{skill_details.get('score', 0):.1f}%")
            
            if skill_details.get('matched'):
                st.write("✅ **Matched Skills:**")
                for skill in skill_details['matched'][:10]:
                    st.write(f"• {skill}")
            
            if skill_details.get('missing'):
                st.write("❌ **Missing Skills:**")
                for skill in skill_details['missing'][:10]:
                    st.write(f"• {skill}")
    
    with col2:
        st.write("#### Experience & Education")
        if comparison_results:
            st.metric("Experience", comparison_results.get('experience_match_status', ''))
            st.metric("Education", comparison_results.get('education_match_status', ''))
            
            st.write("#### AI Recommendations")
            for rec in comparison_results.get('ai_recommendations', []):
                st.info(f"💡 {rec}")
    
    # Smart Recommendations
    if comparison.get('parsed_data'):
        st.divider()
        st.write("### 🎯 Smart Recommendations")
        
        job_data = comparison['parsed_data'].get('job', {})
        resume_data = comparison['parsed_data'].get('resume', {})
        
        # Create ComparisonResult object for recommendations
        saved_results = ComparisonResult(
            match_score=comparison.get('match_score', 0),
            overall_summary=comparison_results.get('overall_summary', ''),
            skill_matches_detailed=comparison_results.get('skill_matches_detailed', {}),
            experience_match_status=comparison_results.get('experience_match_status', ''),
            education_match_status=comparison_results.get('education_match_status', ''),
            ai_recommendations=comparison_results.get('ai_recommendations', []),
            missing_keywords=comparison_results.get('missing_keywords', []),
            ai_confidence_scores=comparison_results.get('ai_confidence_scores', {})
        )
        
        smart_recs = rec_engine.generate_smart_recommendations(job_data, resume_data, saved_results)
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Success Probability", f"{smart_recs['success_probability']:.1f}%")
        with col2:
            st.metric("Timeline", smart_recs['estimated_timeline'])
        with col3:
            st.metric("Skills to Learn", len(smart_recs['skill_roadmap']))
        
        # Show top recommendations
        if smart_recs['skill_roadmap']:
            st.write("**Top Skills to Develop:**")
            for i, skill in enumerate(smart_recs['skill_roadmap'][:3]):
                st.write(f"{i+1}. **{skill['skill']}** - {skill['time_estimate']} ({skill['difficulty']})")
        
        if smart_recs['resume_optimizations']:
            st.write("**Resume Improvements:**")
            for improvement in smart_recs['resume_optimizations'][:2]:
                st.info(f"💡 {improvement['suggestion']}")
    
    # Download option for shared view
    st.divider()
    st.write("### 📄 Download Report")
    
    if st.button("📥 Download PDF Report", use_container_width=True):
        pdf_data = sharing.create_professional_report(comparison)
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_data,
            file_name=f"shared_match_report_{comparison.get('id', 'unknown')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    
    # Navigation back to main app
    st.divider()
    if st.button("🏠 Go to Main App", use_container_width=True):
        st.session_state.current_page = "Dashboard"
        st.rerun()

def render_job_library():
    st.header("💼 Job Library")
    
    jobs = load_all_jobs()
    
    # Add new job
    with st.expander("➕ Add New Job", expanded=not jobs):
        col1, col2 = st.columns([3, 1])
        with col1:
            new_job_text = st.text_area("Paste job description:", height=200)
        with col2:
            st.write("")  # Spacer
            st.write("")
            if st.button("Add Job", disabled=not new_job_text):
                with st.spinner("Processing with AI..."):
                    connected, models = check_ollama_connection()
                    available_models = ['llama3', 'mistral', 'mixtral']
                    
                    if connected and models:
                        # Use available models (up to 2)
                        models_to_use = [m for m in available_models if m in models][:2]
                        
                        # Parse job description with caching
                        parsed, confidence = run_async(cached_multi_model_parse(new_job_text, 'job', models_to_use))
                        st.success(f"Job parsed with {confidence*100:.1f}% confidence!")
                    else:
                        st.error("❌ AI model not available. Please install transformers and torch.")
                        st.stop()
                    
                    # Validate parsed job data
                    job_validation = validator.validate_job(parsed)
                    
                    # Show validation results
                    if not job_validation.is_valid:
                        st.error("❌ Job validation errors:")
                        for error in job_validation.errors:
                            st.error(f"• {error}")
                    if job_validation.warnings:
                        with st.expander("⚠️ Job suggestions"):
                            for warning in job_validation.warnings:
                                st.warning(f"• {warning}")
                    
                    # Use sanitized data
                    parsed = job_validation.sanitized_data
                    parsed['raw_text'] = new_job_text
                    save_job(parsed)
                    st.success("Job added successfully!")
                    st.rerun()
    
    # Display jobs
    if jobs:
        for job in jobs:
            with st.expander(f"{job.get('title', 'Unknown')} - {job.get('company', 'Unknown')}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Location:** {job.get('location', 'Unknown')}")
                    st.write(f"**Type:** {job.get('type', 'Unknown')}")
                    st.write(f"**Experience:** {job.get('experience', 'Not specified')}")
                    st.write(f"**Added:** {job.get('date_added', '')}")
                    
                    if job.get('required_skills'):
                        st.write("**Required Skills:**")
                        st.write(", ".join(job['required_skills'][:10]))
                
                with col2:
                    if st.button("Use in Comparison", key=f"use_job_{job['id']}"):
                        st.session_state.current_page = "New Comparison"
                        st.session_state.selected_job_id = job['id']
                        st.rerun()
                    
                    if st.button("Delete", key=f"delete_job_{job['id']}"):
                        os.remove(JOBS_DIR / f"{job['id']}.json")
                        st.success("Job deleted!")
                        st.rerun()
    else:
        st.info("No jobs in library yet. Add your first job above!")

def render_resume_library():
    st.header("📄 Resume Library")
    
    resumes = load_all_resumes()
    
    # Upload new resume
    with st.expander("➕ Upload New Resume", expanded=not resumes):
        upload_type = st.radio("Upload type:", ["Single File", "Multiple Files"])
        
        if upload_type == "Single File":
            uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx', 'txt'])
        else:
            uploaded_files = st.file_uploader("Choose files", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
        
        if upload_type == "Single File" and uploaded_file:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**File:** {uploaded_file.name}")
                st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
            with col2:
                if st.button("Process Resume"):
                    with st.spinner("Processing with AI..."):
                        try:
                            # Extract text
                            text = extract_text_from_file(uploaded_file)
                            
                            # Parse
                            connected, models = check_ollama_connection()
                            available_models = ['llama3', 'mistral', 'mixtral']
                            
                            if connected and models:
                                # Use available models (up to 2)
                                models_to_use = [m for m in available_models if m in models][:2]
                                
                                # Parse resume with caching
                                parsed, confidence = run_async(cached_multi_model_parse(text, 'resume', models_to_use))
                                st.success(f"Resume parsed with {confidence*100:.1f}% confidence!")
                            else:
                                st.error("❌ AI model not available. Please install transformers and torch.")
                                st.stop()
                            
                            # Validate parsed resume data
                            resume_validation = validator.validate_resume(parsed)
                            
                            # Show validation results
                            if not resume_validation.is_valid:
                                st.error("❌ Resume validation errors:")
                                for error in resume_validation.errors:
                                    st.error(f"• {error}")
                            if resume_validation.warnings:
                                with st.expander("⚠️ Resume suggestions"):
                                    for warning in resume_validation.warnings:
                                        st.warning(f"• {warning}")
                            
                            # Use sanitized data
                            parsed = resume_validation.sanitized_data
                            parsed['raw_text'] = text
                            
                            # Save file
                            file_path = RESUME_FILES_DIR / f"{uuid.uuid4()}_{uploaded_file.name}"
                            with open(file_path, 'wb') as f:
                                f.write(uploaded_file.getvalue())
                            
                            parsed['file_path'] = str(file_path)
                            save_resume(parsed)
                            
                            st.success("Resume uploaded successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
        
        elif upload_type == "Multiple Files" and uploaded_files:
            st.write(f"**Files selected:** {len(uploaded_files)}")
            for file in uploaded_files:
                st.write(f"• {file.name} ({file.size / 1024:.1f} KB)")
            
            if st.button("Process All Resumes"):
                with st.spinner(f"Processing {len(uploaded_files)} resumes with AI..."):
                    try:
                        # Process all files in parallel
                        results = run_async(batch_process_resumes(uploaded_files))
                        
                        successful = 0
                        for i, (parsed, error) in enumerate(results):
                            if parsed:
                                # Save file
                                file_path = RESUME_FILES_DIR / f"{uuid.uuid4()}_{uploaded_files[i].name}"
                                with open(file_path, 'wb') as f:
                                    f.write(uploaded_files[i].getvalue())
                                
                                parsed[0]['raw_text'] = extract_text_from_file(uploaded_files[i])
                                parsed[0]['file_path'] = str(file_path)
                                save_resume(parsed[0])
                                successful += 1
                            else:
                                st.error(f"Failed to process {uploaded_files[i].name}: {error}")
                        
                        st.success(f"Successfully processed {successful}/{len(uploaded_files)} resumes!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Batch processing error: {e}")
    
    # Display resumes
    if resumes:
        for resume in resumes:
            with st.expander(f"{resume.get('name', 'Unknown')} - {resume.get('email', 'Unknown')}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Email:** {resume.get('email', 'Not provided')}")
                    st.write(f"**Phone:** {resume.get('phone', 'Not provided')}")
                    
                    if resume.get('skills'):
                        st.write("**Skills:**")
                        st.write(", ".join(resume['skills'][:15]))
                    
                    if resume.get('education'):
                        st.write(f"**Education:** {resume['education']}")
                
                with col2:
                    if st.button("Use in Comparison", key=f"use_resume_{resume['id']}"):
                        st.session_state.current_page = "New Comparison"
                        st.session_state.selected_resume_id = resume['id']
                        st.rerun()
                    
                    if st.button("Delete", key=f"delete_resume_{resume['id']}"):
                        os.remove(RESUMES_DIR / f"{resume['id']}.json")
                        # Delete file if exists
                        if resume.get('file_path') and os.path.exists(resume['file_path']):
                            os.remove(resume['file_path'])
                        st.success("Resume deleted!")
                        st.rerun()
    else:
        st.info("No resumes in library yet. Upload your first resume above!")

def render_settings():
    st.header("⚙️ Settings")
    
    settings = st.session_state.settings
    
    # General settings
    st.subheader("General Settings")
    col1, col2 = st.columns(2)
    with col1:
        settings['theme'] = st.selectbox("Theme:", ["light", "dark"], 
            index=["light", "dark"].index(settings.get('theme', 'light')))
    with col2:
        settings['default_view'] = st.selectbox("Default view:", ["grid", "list"],
            index=["grid", "list"].index(settings.get('default_view', 'grid')))
    
    st.divider()
    
    # AI Configuration
    st.subheader("AI Configuration")
    
    # Check SmolLM2 availability
    connected, models = check_ollama_connection()
    
    if connected:
        st.success("✅ SmolLM2 is available")
        col1, col2 = st.columns(2)
        with col1:
            settings['model'] = st.selectbox("Model:", models,
                index=models.index(settings.get('model', 'llama3')) if settings.get('model', 'llama3') in models else 0)
        with col2:
            settings['temperature'] = st.slider("Temperature:", 0.0, 1.0, 
                value=settings.get('temperature', 0.1), step=0.1)
    else:
        st.error("❌ SmolLM2 is not available. Please install transformers and torch.")
    
    st.divider()
    
    # Scoring weights
    st.subheader("Scoring Weights")
    st.info("Adjust the importance of each factor in the matching algorithm")
    
    col1, col2 = st.columns(2)
    with col1:
        settings['skill_weight'] = st.slider("Skills Weight (%)", 0, 100, 
            value=settings.get('skill_weight', 40))
        settings['experience_weight'] = st.slider("Experience Weight (%)", 0, 100,
            value=settings.get('experience_weight', 30))
    with col2:
        settings['education_weight'] = st.slider("Education Weight (%)", 0, 100,
            value=settings.get('education_weight', 15))
        settings['other_weight'] = st.slider("Other Factors Weight (%)", 0, 100,
            value=settings.get('other_weight', 15))
    
    # Ensure weights sum to 100
    total_weight = sum([settings['skill_weight'], settings['experience_weight'], 
                       settings['education_weight'], settings['other_weight']])
    if total_weight != 100:
        st.warning(f"Weights must sum to 100%. Current total: {total_weight}%")
    
    settings['llm_blend'] = st.slider("LLM Influence (%)", 0, 100,
        value=settings.get('llm_blend', 30),
        help="How much the AI model influences the final score vs rule-based scoring")
    
    st.divider()
    
    # Export settings
    st.subheader("Export Settings")
    settings['export_format'] = st.selectbox("Default export format:", 
        ["pdf", "json", "csv"],
        index=["pdf", "json", "csv"].index(settings.get('export_format', 'pdf')))
    
    st.divider()
    
    # Cache Management
    st.subheader("Cache Management")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cache Hit Rate", f"{cache.hit_rate*100:.1f}%")
    with col2:
        st.metric("Cache Hits", cache.cache_stats["hits"])
    with col3:
        st.metric("Cache Misses", cache.cache_stats["misses"])
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Cache", type="secondary"):
            cache.clear_cache()
            st.success("Cache cleared successfully!")
            st.rerun()
    with col2:
        cache_age = st.slider("Cache Age Limit (hours)", 1, 168, 
            value=24, help="How long to keep cached results")
    
    st.divider()
    
    # Validation Settings
    st.subheader("🔍 Validation Settings")
    st.info("Configure data validation rules and thresholds")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Resume Validation:**")
        min_skills = st.slider("Minimum Skills", 1, 10, value=3, 
            help="Minimum number of skills required")
        max_skills = st.slider("Maximum Skills", 20, 100, value=50, 
            help="Maximum number of skills before warning")
        
        st.write("**Job Validation:**")
        min_job_skills = st.slider("Minimum Job Skills", 1, 10, value=2, 
            help="Minimum number of required skills for jobs")
        max_job_skills = st.slider("Maximum Job Skills", 20, 50, value=30, 
            help="Maximum number of required skills before warning")
    
    with col2:
        st.write("**Keyword Detection:**")
        keyword_threshold = st.slider("Keyword Stuffing Threshold (%)", 1, 10, value=2, 
            help="Percentage threshold for detecting keyword stuffing")
        
        st.write("**Validation Behavior:**")
        strict_validation = st.checkbox("Strict Validation Mode", value=False,
            help="Treat warnings as errors in strict mode")
        
        auto_sanitize = st.checkbox("Auto-sanitize Data", value=True,
            help="Automatically clean and sanitize input data")
    
    # Save validation settings
    settings['validation'] = {
        'min_skills': min_skills,
        'max_skills': max_skills,
        'min_job_skills': min_job_skills,
        'max_job_skills': max_job_skills,
        'keyword_threshold': keyword_threshold,
        'strict_validation': strict_validation,
        'auto_sanitize': auto_sanitize
    }
    
    # Save button
    if st.button("Save Settings", type="primary"):
        save_settings(settings)
        st.success("Settings saved successfully!")
        st.rerun()

def render_help():
    st.header("❓ Help & Support")
    
    # User guide
    with st.expander("📖 User Guide", expanded=True):
        st.markdown("""
        ### Getting Started
        
        1. **Upload your resume** to the Resume Library
        2. **Add job descriptions** to the Job Library
        3. **Create a comparison** to see how well they match
        4. **Review recommendations** to improve your chances
        
        ### Key Features
        
        - **Advanced AI Parsing**: Multi-model consensus parsing with confidence scores
        - **Smart Caching**: Intelligent caching system for faster processing
        - **Smart Parsing**: AI-powered extraction of key information
        - **Detailed Analysis**: Skills, experience, and education matching
        - **Scoring System**: Customizable weights for different factors
        - **Smart Recommendations**: Personalized action plans with skill roadmaps
        - **Batch Processing**: Process multiple files simultaneously with comprehensive analysis
        - **Gamification**: Earn achievements, level up, and track your progress
        - **Secure Sharing**: Share results via secure links and QR codes
        - **Professional Reports**: Download PDF reports for presentations
        
        ### Tips for Best Results
        
        - Use clear, well-formatted documents
        - Include all relevant skills in your resume
        - Keep job descriptions complete and detailed
        - Review and edit parsed data for accuracy
        - Use natural language - avoid keyword stuffing
        - Ensure contact information is accurate
        """)
    
    # FAQs
    with st.expander("❓ Frequently Asked Questions"):
        st.markdown("""
        **Q: How accurate is the parsing?**
        A: The AI parsing is highly accurate, but we recommend reviewing the extracted data.
        
        **Q: Can I edit the parsed information?**
        A: Yes! Step 2 of the comparison wizard allows full editing.
        
        **Q: What file formats are supported?**
        A: PDF, DOCX, and TXT files are supported.
        
        **Q: How is the match score calculated?**
        A: The score combines multiple factors with customizable weights in Settings.
        
        **Q: What is the caching system?**
        A: The app caches parsing results to speed up repeated processing of similar documents.
        
        **Q: How long are cached results kept?**
        A: By default, cached results are kept for 24 hours, but this can be adjusted in Settings.
        
        **Q: How accurate are the skill recommendations?**
        A: Recommendations are based on skill relationships and learning patterns, but individual results may vary.
        
        **Q: Can I track my progress on recommended skills?**
        A: Yes! Use the Progress Tracking tab in the Recommendations page to monitor your skill development.
        
        **Q: How do I share my comparison results?**
        A: Use the "Create Share Link" button to generate a secure link that expires in 48 hours, or download a PDF report.
        
        **Q: Are shared links secure?**
        A: Yes! Links use secure tokens and expire automatically. No personal data is stored in the links.
        
        **Q: What is data validation?**
        A: The app validates parsed data for accuracy, completeness, and potential issues like keyword stuffing.
        
        **Q: How does keyword stuffing detection work?**
        A: The system analyzes word frequency and flags unnatural repetition that might hurt your chances.
        
        **Q: Can I customize validation rules?**
        A: Yes! Go to Settings > Validation Settings to adjust thresholds and validation behavior.
        
        **Q: What is data sanitization?**
        A: The app automatically cleans input data by removing HTML tags and escaping special characters for security.
        
        **Q: How does batch processing work?**
        A: Upload multiple job descriptions and resumes to compare all combinations at once. Perfect for recruiters and HR professionals.
        
        **Q: What file formats are supported for batch processing?**
        A: PDF, DOCX, and TXT files are supported for both job descriptions and resumes in batch mode.
        
        **Q: How long does batch processing take?**
        A: Processing time depends on the number of files and their complexity. The app shows real-time progress updates.
        
        **Q: Can I download batch processing results?**
        A: Yes! You can download both detailed results (CSV) and summary statistics for further analysis.
        
        **Q: How does the gamification system work?**
        A: Earn XP by completing comparisons, achieving high scores, and matching skills. Level up and unlock achievements!
        
        **Q: What are achievements?**
        A: Achievements are badges you earn for specific accomplishments like high scores, consistency, or skill mastery.
        
        **Q: How do I level up?**
        A: Gain XP by completing comparisons, achieving high match scores, and matching many skills. Higher levels require more XP.
        
        **Q: Can I see my progress towards achievements?**
        A: Yes! Visit the Achievements page to see your earned badges and progress towards new ones.
        """)
    
    # Video tutorials (mock)
    with st.expander("🎥 Video Tutorials"):
        st.info("Video tutorials coming soon!")
        st.write("• Getting Started (5 min)")
        st.write("• Advanced Features (10 min)")
        st.write("• Tips & Tricks (7 min)")
    
    # Contact form
    with st.expander("📧 Contact Support"):
        with st.form("contact_form"):
            name = st.text_input("Name:")
            email = st.text_input("Email:")
            subject = st.selectbox("Subject:", 
                ["Bug Report", "Feature Request", "General Question", "Other"])
            message = st.text_area("Message:", height=150)
            
            if st.form_submit_button("Send Message"):
                st.success("Message sent! We'll get back to you within 24 hours.")

# Advanced CSS with animations and modern styling
def load_advanced_css():
    st.markdown("""
    <style>
    /* Animated gradient background */
    .stApp {
        background: linear-gradient(-45deg, #f0f2f6, #e8ebf0, #f0f2f6, #ffffff);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.25);
    }
    
    /* Animated match score */
    .match-score {
        font-size: 72px;
        font-weight: bold;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Smooth transitions */
    * {
        transition: all 0.3s ease;
    }
    
    /* Modern buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        box-shadow: 0 4px 15px 0 rgba(31, 38, 135, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(31, 38, 135, 0.4);
    }
    
    /* Progress bars with animation */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
        animation: shimmer 2s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    
    /* Enhanced metric styling */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.25);
    }
    
    /* Enhanced sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    /* Enhanced expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.9);
        transform: translateY(-1px);
    }
    
    /* Enhanced info box styling */
    .stAlert {
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    /* Main container styling */
    .main {
        padding: 2rem;
        animation: fadeIn 0.5s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Skeleton loader styling */
    @keyframes skeleton-loading {
        0% { background-position: -200px 0; }
        100% { background-position: calc(200px + 100%) 0; }
    }
    
    .skeleton-box {
        display: inline-block;
        height: 1em;
        position: relative;
        overflow: hidden;
        background-color: #DDDBDD;
        background-image: linear-gradient(90deg, #DDDBDD 0px, #F5F5F5 40px, #DDDBDD 80px);
        background-size: 200px 100%;
        background-repeat: no-repeat;
        animation: skeleton-loading 1.2s ease-in-out infinite;
        border-radius: 4px;
        margin: 5px 0;
    }
    
    /* Chart container styling */
    .js-plotly-plot {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.9);
        transform: translateY(-1px);
    }
    
    /* AI Thinking Animation */
    .ai-thinking-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2px;
        border-radius: 10px;
        margin: 10px 0;
        animation: thinking-glow 2s ease-in-out infinite;
    }
    
    @keyframes thinking-glow {
        0%, 100% { box-shadow: 0 0 5px rgba(102, 126, 234, 0.5); }
        50% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.8); }
    }
    
    .ai-thought-stream {
        font-family: 'Courier New', monospace;
        font-size: 14px;
        color: #333;
        white-space: pre-wrap;
        max-height: 200px;
        overflow-y: auto;
        background: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border-left: 3px solid #667eea;
    }
    
    .token-counter {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 15px 20px;
        border-radius: 30px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        font-weight: 600;
        z-index: 1000;
    }
    
    .stage-indicator {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 8px 16px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 500;
    }
    
    .thinking-pulse {
        width: 8px;
        height: 8px;
        background: white;
        border-radius: 50%;
        animation: pulse 1s ease-in-out infinite;
    }
    </style>
    """, unsafe_allow_html=True)

# Add skeleton loaders during processing
def show_skeleton_loader():
    skeleton_html = """
    <div class="skeleton-loader">
        <div class="skeleton-box" style="width: 80%; height: 20px;"></div>
        <div class="skeleton-box" style="width: 60%; height: 20px;"></div>
        <div class="skeleton-box" style="width: 70%; height: 20px;"></div>
        <div class="skeleton-box" style="width: 50%; height: 20px;"></div>
    </div>
    """
    st.markdown(skeleton_html, unsafe_allow_html=True)

# Animated transitions between pages
def page_transition():
    st.markdown("""
    <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .main > div {
            animation: fadeIn 0.5s ease-out;
        }
    </style>
    """, unsafe_allow_html=True)

# Main app
def main():
    st.set_page_config(
        page_title="Job-Resume Matcher",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Check requirements first
    check_requirements()
    
    # Initialize components with error handling
    try:
        load_advanced_css()
        page_transition()
        init_session_state()
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        st.info("Please refresh the page or check your installation.")
        return
    
    # Check for shared link
    query_params = st.query_params
    if 'share' in query_params:
        token = query_params['share']
        shared_comparison = sharing.get_shared_comparison(token)
        
        if shared_comparison:
            st.header("📊 Shared Comparison Results")
            st.info("This is a shared comparison link. The link will expire in 48 hours.")
            
            # Display shared comparison
            render_shared_comparison(shared_comparison)
        else:
            st.error("This shared link has expired or is invalid.")
            st.button("Go to Dashboard", on_click=lambda: setattr(st.session_state, 'current_page', 'Dashboard'))
            return
    
    render_sidebar()
    
    # Route to appropriate page
    page = st.session_state.current_page
    
    if page == "Dashboard":
        render_dashboard()
    elif page == "New Comparison":
        render_new_comparison()
    elif page == "Batch Processing":
        render_batch_processing()
    elif page == "My Comparisons":
        render_comparisons()
    elif page == "Job Library":
        render_job_library()
    elif page == "Resume Library":
        render_resume_library()
    elif page == "Recommendations":
        render_recommendations()
    elif page == "Achievements":
        render_achievements()
    elif page == "Settings":
        render_settings()
    elif page == "Help/Support":
        render_help()

if __name__ == "__main__":
    main()