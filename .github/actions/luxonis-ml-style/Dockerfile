# Use an image that has the necessary dependencies
FROM python:3.10-slim

# Install pre-commit
RUN pip install pre-commit

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y git

# Copy the shared pre-commit configuration and the entrypoint script
COPY pre-commit-config.yaml /.pre-commit-config.yaml
COPY entrypoint.sh /entrypoint.sh

# Set the entrypoint to the entrypoint script
ENTRYPOINT ["/entrypoint.sh"]
