#!/usr/bin/env bash

set -e
set -x

mypy "power" "tests"
flake8 "power" "tests" --ignore=E501,W503,E203,E402,E704
black "power" "tests" --check -l 80
