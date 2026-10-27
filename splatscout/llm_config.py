"""Large-model endpoint settings, all from environment variables.

  OPENAI_API_KEY    API key (required for the API backends)
  OPENAI_BASE_URL   base URL of an OpenAI-compatible endpoint (default https://api.openai.com/v1)
  AGENT_MODEL       model name (default gpt-5.6)
The Responses backend (default) uses <base>/responses; the chat backend uses <base>/chat/completions.
"""
import os

_BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
LLM_URL = _BASE + "/chat/completions"
LLM_MODEL = os.environ.get("AGENT_MODEL", "gpt-5.6")
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "")
