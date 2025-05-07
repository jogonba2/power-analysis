#!/usr/bin/env bash

set -e
set -x

mypy "power"
flake8 "power" --ignore=E501,W503,E203,E402,E704
black "power" --check -l 80
