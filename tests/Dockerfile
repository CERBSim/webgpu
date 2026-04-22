ARG BASE_IMAGE=webgpu-base
FROM ${BASE_IMAGE}

WORKDIR /app
COPY tests/ tests/

CMD ["pytest", "tests/", "-v", "--tb=short"]