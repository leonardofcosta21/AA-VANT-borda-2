# Campaign shortcuts. Every target is a thin wrapper over run_campaign.py,
# which stays the single entry point -- the Makefile must never become a
# second place where experiment parameters live.

PY      ?= python
RESULTS ?= results

.PHONY: help audit plan smoke test e1 e2 e3 e4 e5 e6 priority1 all report status matrix clean-mock

help:
	@echo "Pre-flight"
	@echo "  make audit       check which datasets are present and usable"
	@echo "  make plan        planned runs and a wall-clock estimate"
	@echo "  make smoke       validate the whole pipeline with the mock backend (no GPU)"
	@echo "  make test        unit tests for metrics, statistics and expansion"
	@echo ""
	@echo "Run (resumable; safe to interrupt and repeat)"
	@echo "  make priority1   E1, E2a, E2b, E6 -- everything the defence depends on"
	@echo "  make e1 .. e6    one block at a time"
	@echo "  make all         the entire campaign"
	@echo ""
	@echo "Report"
	@echo "  make status      progress per block"
	@echo "  make report      rebuild every table and figure from stored records"

audit:
	$(PY) scripts/audit_datasets.py --json $(RESULTS)/dataset_audit.json

plan:
	$(PY) run_campaign.py plan

smoke:
	$(PY) run_campaign.py --output $(RESULTS)/_mock run --all --mock
	$(PY) run_campaign.py --output $(RESULTS)/_mock report --tables tables/_mock --figures figures/_mock
	@echo "Mock campaign complete. Inspect tables/_mock and figures/_mock, then 'make clean-mock'."

test:
	$(PY) -m pytest tests/ -q

e1:
	$(PY) run_campaign.py run --block E1_main_grid
e2:
	$(PY) run_campaign.py run --block E2a_shift_baseline E2b_shift_real E2c_shift_intensity
e3:
	$(PY) run_campaign.py run --block E3_uncertainty_methods
e4:
	$(PY) run_campaign.py run --block E4_sensitivity
e5:
	$(PY) run_campaign.py run --block E5_long_horizon
e6:
	$(PY) run_campaign.py run --block E6_operational_profiling

priority1:
	$(PY) run_campaign.py run --block E1_main_grid E2a_shift_baseline \
	                                  E2b_shift_real E6_operational_profiling

all:
	$(PY) run_campaign.py run --all

status:
	$(PY) run_campaign.py status

report:
	$(PY) run_campaign.py report

matrix:
	$(PY) run_campaign.py matrix

clean-mock:
	rm -rf $(RESULTS)/_mock tables/_mock figures/_mock
