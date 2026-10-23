"""Regenerate standalone chart files from the persisted metrics JSON."""

from cafe_agent.cpu.settings import local_settings
from cafe_agent.reporting.charts import make_charts

if __name__ == "__main__":
    settings = local_settings()
    make_charts(settings.report.directory / "metrics.json", settings.report.charts_directory)
