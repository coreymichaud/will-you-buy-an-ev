.PHONY: help install all


# ============ MISC COMMANDS ============

help:  # Shows a help message, and is the default `make` target
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  help             Shows this help message"
	@echo "  install          Syncs the environment"
	@echo "  download         Downloads the data from Kaggle"
	@echo "  train            Trains the model"
	@echo "  inference        Runs inference on the model"
	@echo "  all              Runs the full ML pipeline"

install:  # Syncs the environment
	uv sync


# ============ DATA COMMANDS ============

download:  # Downloads the data from Kaggle
	uv run python -m will_you_buy_an_ev.data.download

train: download  # Trains the model
	uv run python -m will_you_buy_an_ev.model.train

inference: train  # Runs inference on the model
	uv run python -m will_you_buy_an_ev.model.inference


# ============ FULL PIPELINE COMMAND ============

all: install inference # Runs the full ML pipeline