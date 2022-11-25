.PHONY: clean data lint requirements

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = socio-ethnic_segregation
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	conda install -c conda-forge geopandas pygeos -y
	conda install --file requirements.txt -y

## Make Dataset
data:
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/interim
	cp data/raw/income_tax_record_1880.csv data/interim/
	$(PYTHON_INTERPRETER) src/features/build_features.py data/interim data/processed \
	--n_clusters 12 --seed 42
	cp data/interim/water_1913.gpkg data/processed/

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint and style
lint:
	black src notebooks
	flake8 src

## Set up python interpreter environment
create_environment:
	conda create -c conda-forge "pymc>=4" --name $(PROJECT_NAME) -y
	@echo ">>> New conda env created."

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Train models
train:
	$(PYTHON_INTERPRETER) src/models/train_model.py data/processed models reports/figures \
	--seed 42 --prior_samples 100 --draws 1000 --tune 1000 --target_accept 0.95

## Save requirements to file
save:
	conda list --export > requirements.txt

## Delete conda environment
delete:
	conda deactivate
	conda env remove -n $(PROJECT_NAME)

## Draw figures for reporting
figures: ./reports/figures/plate_diagram.svg
	rsvg-convert ./reports/figures/plate_diagram.svg -f png -o ./reports/figures/plate_diagram.png -d 600 -p 600
	$(PYTHON_INTERPRETER) src/visualization/visualize.py data/processed models reports/figures
	$(PYTHON_INTERPRETER) src/visualization/flowchart.py reports/figures
	rsvg-convert ./reports/figures/flowchart.svg -f png -o ./reports/figures/flowchart.png -d 600 -p 60

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
