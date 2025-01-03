install:
	poetry install --no-root

test:
	poetry run pytest ./tests -vv

update:
	poetry update

format:
	poetry run ruff format .
	poetry run ruff check . --fix

check:
	poetry run ruff format --check .

studio-train:   ## Run training on Lightning AI Studio
	lightning run train.py

studio-sync:   ## Sync Lightning AI Studio
	lightning upload-project --path .