PYTHON ?= python3
IN ?= data/tickets_valid.jsonl
OUT ?= data/pred.jsonl
WORKERS ?= 8
PORT ?= 8000
STRIP_LABELS ?= 1
# Режим пакетного прогона: full | strip-labels | text-only.
# STRIP_LABELS=0 оставлен для обратной совместимости и включает режим full.
MODE ?= $(if $(filter 0 false no,$(STRIP_LABELS)),full,strip-labels)

.PHONY: help predict run eval check-citations bench test clean-data docker-build docker-run

help:
	@echo "Grand Mobile Support Copilot — Makefile targets:"
	@echo "  make predict IN=tickets.jsonl OUT=pred.jsonl [MODE=full|strip-labels|text-only] [WORKERS=8]"
	@echo "                                                               - Run batch prediction on JSONL"
	@echo "  make run                                                     - Start FastAPI server on port $(PORT)"
	@echo "  make eval                                                    - Run evaluation & generate manual markup sample"
	@echo "  make check-citations                                         - Verify all cited rules on 1,000 valid tickets"
	@echo "  make bench                                                   - Run latency & cost benchmark (cached & uncached)"
	@echo "  make test                                                    - Run test suite (pytest if installed, else unittest)"
	@echo "  make clean-data                                              - Remove all case data files from data/"
	@echo "  make docker-build                                            - Build Docker image grand-mobile-copilot:latest"
	@echo "  make docker-run                                              - Run Docker container mounting ./data into /app/data:ro"

predict:
	$(PYTHON) scripts/predict.py --input "$(IN)" --output "$(OUT)" --workers $(WORKERS) --mode $(MODE)

run:
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port $(PORT)

eval:
	$(PYTHON) eval/run_eval.py

check-citations:
	$(PYTHON) eval/check_citations.py

bench:
	$(PYTHON) eval/bench.py

test:
	@if $(PYTHON) -c "import pytest" 2>/dev/null; then \
		$(PYTHON) -m pytest -v; \
	else \
		$(PYTHON) -m unittest discover -s tests -p "test_*.py" -v; \
	fi

clean-data:
	rm -rf data/*.jsonl data/*.jsonl.* data/rules.json data/taxonomy.json data/manual_annotation_100.jsonl data/pred*.jsonl data/*.json

docker-build:
	docker build -t grand-mobile-copilot:latest .

docker-run:
	docker run --rm -p $(PORT):8000 --env-file .env -v "$$PWD/data:/app/data:ro" grand-mobile-copilot:latest
