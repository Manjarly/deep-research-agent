"""Configuration settings for the Autonomous Deep Research Agent."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


class Settings:
    """Application settings and API keys."""

    # Google Gemini Settings
    GOOGLE_API_KEY: str = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or ""
    )
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    MODEL_TEMPERATURE: float = float(os.getenv("MODEL_TEMPERATURE", "0.2"))

    # Optional alternative LLM keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Search Configuration
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    SEARCH_PROVIDER: str = os.getenv("SEARCH_PROVIDER", "duckduckgo")  # 'duckduckgo' or 'tavily'
    MAX_SEARCH_RESULTS_PER_QUERY: int = int(os.getenv("MAX_SEARCH_RESULTS_PER_QUERY", "4"))
    MAX_SCRAPE_CHARS: int = int(os.getenv("MAX_SCRAPE_CHARS", "6000"))

    # Agent Loop Controls
    MAX_RESEARCH_LOOPS: int = int(os.getenv("MAX_RESEARCH_LOOPS", "2"))

    # Output directory
    REPORTS_DIR: Path = ROOT_DIR / "reports"

    @classmethod
    def validate_keys(cls, provider: str = "gemini") -> bool:
        """Check if required API keys are configured."""
        if provider == "gemini":
            return bool(cls.GOOGLE_API_KEY and cls.GOOGLE_API_KEY != "your_gemini_api_key_here")
        elif provider == "openai":
            return bool(cls.OPENAI_API_KEY)
        elif provider == "anthropic":
            return bool(cls.ANTHROPIC_API_KEY)
        return True


# Ensure reports directory exists
Settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
config = Settings()
