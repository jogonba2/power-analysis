#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place "power" "tests" --exclude=__init__.py
isort "power" "tests"
black "power" "tests" -l 80
