"""Text pipeline components for FinTok."""

from .chunk_selector import get_context, build_context_groups
from .fact_extractor import extract_facts
from .fact_checker import select_fact
from .script_generator import generate_script, build_video_json
