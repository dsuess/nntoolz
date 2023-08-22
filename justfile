# Needs at least just 1.9.0

test:
    poetry run pytest

lint:
    poetry run isort --check nntoolz tests
    poetry run black --check nntoolz tests
    poetry run ruff check nntoolz

fix:
   poetry run isort nntoolz tests
   poetry run black nntoolz tests
   poetry run ruff check --fix nntoolz
