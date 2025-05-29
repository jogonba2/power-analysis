#!/usr/bin/env bash

set -e
set -x

mypy "power" "tests" "examples"
flake8 "power" "tests" "examples" --ignore=E501,W503,E203,E402,E704
black "power" "tests" "examples" --check -l 80
