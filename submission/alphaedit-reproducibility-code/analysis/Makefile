# Paper Figure & Table Generation
#
# Usage:
#   make all          Generate everything
#   make fig2         Figure 2: Reproduction + long-horizon boundary
#   make fig3         Figure 3: Local plasticity vs global retention
#   make fig4         Figure 4: Stream concentration & interference
#   make fig5         Figure 5: Attribution correction
#   make interference Run interference panel analysis (requires keys)
#   make mechanism    Ordering mechanism (interference distillation)
#   make tables       All tables + paper_numbers.json
#   make appendix     Appendix figures (A1, A3, A8, capability)
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

.PHONY: all fig2 fig3 fig4 fig5 interference mechanism tables appendix appendix-capability check clean

all: fig2 fig3 interference fig4 fig5 mechanism tables

fig2:
	@echo "── Figure 2: Reproduction & Long-Horizon ──"
	$(PYTHON) -m analysis.fig2_reproduction --output-dir $(OUTPUT)

fig3:
	@echo "── Figure 3: Local Plasticity vs Global Retention ──"
	$(PYTHON) -m analysis.fig3_plasticity --output-dir $(OUTPUT)

interference:
	@echo "── Interference Panel Analysis ──"
	$(PYTHON) -m analysis.interference_panel --output-dir $(OUTPUT) --keys-dir $(KEYS)

fig4:
	@echo "── Figure 4: Stream Concentration & Interference ──"
	$(PYTHON) -m analysis.fig4_mechanism --output-dir $(OUTPUT)

fig5: interference
	@echo "── Figure 5: Attribution Correction ──"
	$(PYTHON) -m analysis.fig5_attribution --output-dir $(OUTPUT)

mechanism:
	@echo "── Ordering Mechanism (Interference Distillation) ──"
	$(PYTHON) -m analysis.distill_interference --output-dir $(OUTPUT)

tables:
	@echo "── Tables & Paper Numbers ──"
	$(PYTHON) -m analysis.paper_tables --output-dir $(OUTPUT)

appendix: appendix-capability
	@echo "── Appendix Figures ──"
	$(PYTHON) -m analysis.appendix_figures --output-dir $(APPENDIX)

appendix-capability:
	@echo "── Appendix: Capability Probe ──"
	$(PYTHON) -m analysis.appendix_capability --output-dir $(APPENDIX)

check:
	@echo "── Data Availability Report ──"
	$(PYTHON) -m analysis.loaders

clean:
	rm -f $(OUTPUT)/fig*.png $(OUTPUT)/fig*.pdf
	rm -f $(OUTPUT)/table*.csv $(OUTPUT)/paper_numbers.json
	rm -f $(APPENDIX)/*.png $(APPENDIX)/*.pdf
