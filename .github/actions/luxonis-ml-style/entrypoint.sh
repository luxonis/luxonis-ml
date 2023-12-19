#!/bin/sh -l

pre-commit install
pre-commit run --all-files

exit $?
