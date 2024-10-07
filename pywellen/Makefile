.PHONY: help develop wheel test 

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


deps:  ## Installs the required dependencies to develop pywellen -- we use pyenv to manage python versions
	python3 -m venv dev && source dev/bin/activate && pip install pytest maturin 


develop: ## Builds pywellen, and installs it into the current virtual env
	maturin develop 

wheel: ## Builds the pywellen wheel and is placed in dist/
	maturin build  --out dist

test: develop ## Builds and tests pywellen via pytest
	pytest tests/
