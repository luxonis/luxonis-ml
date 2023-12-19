#!/bin/sh -l

# Install pre-commit and run it
pre-commit install
pre-commit run --all-files

# Exit with the status of the last command run
exit $?
