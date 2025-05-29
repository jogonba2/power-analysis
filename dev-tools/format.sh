#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place "power" "tests" "examples" --exclude=__init__.py
isort "power" "tests" "examples"
black "power" "tests" "examples" -l 80
