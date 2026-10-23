"""Regenerate the README results block and Markdown tables from saved metrics."""

from cafe_agent.cpu.settings import local_settings
from cafe_agent.reporting.text import write_text

if __name__ == "__main__":
    settings = local_settings()
    write_text(settings.report.directory / "metrics.json", settings.report.directory)
