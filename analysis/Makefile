# Paper Figure & Table Generation
#
# Usage:
#   make all          Generate everything
#   make fig1         Figure 1: Reproduction + long-horizon
#   make fig2         Figure 2: Anatomy of forgetting
#   make fig5         Figure 5: Per-edit interference (key geometry)
#   make interference Run interference panel analysis (requires keys)
#   make tables       All tables + paper_numbers.json
#   make appendix     Appendix figures (A1, A3, A8)
#   make check        Verify data availability
#   make clean        Remove generated outputs
#
# All outputs go to results/figures/paper/ (or appendix/).
# Run from the repo root: make -C analysis all

PYTHON := PYTHONPATH=.. uv run python
OUTPUT := ../results/figures/paper
APPENDIX := ../results/figures/appendix
KEYS := ../results/key_vectors
MATCHED := ../results/matched_ordering

.PHONY: all fig1 fig2 fig5 interference tables appendix check clean validate_ordering

all: fig1 fig2 interference fig5 tables

fig1:
	@echo "── Figure 1: Reproduction & Long-Horizon ──"
	$(PYTHON) -m analysis.fig1_reproduction --output-dir $(OUTPUT)

fig2:
	@echo "── Figure 2: Anatomy of Forgetting ──"
	$(PYTHON) -m analysis.fig2_forgetting --output-dir $(OUTPUT)

interference:
	@echo "── Interference Panel Analysis ──"
	$(PYTHON) -m analysis.interference_panel --output-dir $(OUTPUT) --keys-dir $(KEYS)

fig5: interference
	@echo "── Figure 5: Per-Edit Interference ──"
	$(PYTHON) -m analysis.fig5_interference --output-dir $(OUTPUT)

tables:
	@echo "── Tables & Paper Numbers ──"
	$(PYTHON) -m analysis.paper_tables --output-dir $(OUTPUT)

appendix:
	@echo "── Appendix Figures ──"
	$(PYTHON) -m analysis.appendix_figures --output-dir $(APPENDIX)

validate_ordering:
	@echo "── Matched Ordering: Generate & Validate ──"
	$(PYTHON) src/datasets/generate_orderings.py --seed 42


check:
	@echo "── Data Availability Report ──"
	$(PYTHON) -m analysis.loaders

clean:
	rm -f $(OUTPUT)/fig*.png $(OUTPUT)/fig*.pdf
	rm -f $(OUTPUT)/table*.csv $(OUTPUT)/paper_numbers.json
	rm -f $(APPENDIX)/*.png $(APPENDIX)/*.pdf
