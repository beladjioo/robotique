# Commandes usuelles du monorepo. `make help` pour la liste.
PYTHON ?= python3
PY_PROJECTS := manipulation navigation estimation commande vision drones locomotion essaims apprentissage planification
SRC_PATH := $(subst $(eval) ,:,socle/src $(addprefix projets/,$(addsuffix /src,$(PY_PROJECTS))))
BUILD_DIR ?= build/embarque

.PHONY: help install test test-python test-embarque lint format demos clean

help:
	@echo "install        dépendances de dev + toutes les branches Python en mode éditable"
	@echo "test           tests Python (pytest) et C++ (ctest)"
	@echo "lint / format  vérification et mise en forme du code Python (ruff)"
	@echo "demos          exécute toutes les démos"
	@echo "clean          supprime les fichiers générés"

install:
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e socle $(addprefix -e projets/,$(PY_PROJECTS))

test: test-python test-embarque

test-python:
	$(PYTHON) -m pytest

test-embarque:
	cmake -S projets/embarque -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR)
	ctest --test-dir $(BUILD_DIR) --output-on-failure

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

demos:
	@for demo in projets/*/examples/*.py; do \
		echo "=== $$demo"; \
		PYTHONPATH=$(SRC_PATH) $(PYTHON) $$demo || exit 1; \
	done

clean:
	rm -rf build outputs .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
