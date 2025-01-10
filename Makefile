install:
	poetry install --no-root

test:
	poetry run pytest ./tests -vv
	@if [ -d "webapp" ]; then \
		cd webapp && ./mvnw test; \
	else \
		echo "Directory 'webapp' does not exist!"; \
	fi

update:
	poetry update

format:
	poetry run ruff format .
	poetry run ruff check . --fix

check:
	poetry run ruff format --check .

train:
	poetry run python -m magical_drones.trainer