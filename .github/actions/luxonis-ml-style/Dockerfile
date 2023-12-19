# Use an image that has the necessary dependencies
FROM python:3.10-slim

# Install pre-commit
RUN pip install pre-commit

# Copy the shared pre-commit configuration and the entrypoint script
COPY .pre-commit-config.yaml /.pre-commit-config.yaml
COPY .github/actions/luxonis-ml-style/entrypoint.sh /entrypoint.sh

# Set the entrypoint to the entrypoint script
ENTRYPOINT ["/entrypoint.sh"]
