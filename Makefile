.PHONY: assets clean histogram performance popularity

assets: histogram performance popularity

histogram: assets/frameworks_histogram.svg

performance: assets/performance_comparison.png

popularity: assets/frameworks_trends.png

assets/frameworks_histogram.svg: scripts/generate_histogram.py
	uv run python scripts/generate_histogram.py

assets/performance_comparison.png: scripts/plot_performance.py assets/performance.csv
	uv run python scripts/plot_performance.py

assets/frameworks_trends.png: scripts/plot_popularity.py assets/popularity.csv
	uv run python scripts/plot_popularity.py

clean:
	rm -f assets/frameworks_histogram.svg assets/performance_comparison.png assets/frameworks_trends.png
