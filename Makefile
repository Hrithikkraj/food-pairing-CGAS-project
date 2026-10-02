# Makefile for Flavor-Nutrition Pairing Project Pipeline

.PHONY: all data analysis report test clean

all: data analysis report

# S1 to S4: Ingestion, Filtering, Matching, and Nutrient Profiling
data:
	python run_pipeline.py --stage S1
	python run_pipeline.py --stage S2
	python run_pipeline.py --stage S3
	python run_pipeline.py --stage S4

# S5 to S6: Pair Table Computation and Statistical Analyses
analysis:
	python run_pipeline.py --stage S5
	python run_pipeline.py --stage S6

# S7 to S8: Visualizations, Dashboard, and Packaging
report:
	python run_pipeline.py --stage S7

# Run all unit tests
test:
	python -m unittest discover -s tests -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
