#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place "power" --exclude=__init__.py
isort "power"
black "power" -l 80
