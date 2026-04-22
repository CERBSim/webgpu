ARG BASE_IMAGE=webgpu-base
FROM ${BASE_IMAGE}

WORKDIR /app

COPY pyproject.toml /tmp/webgpu_build/
COPY webgpu/ /tmp/webgpu_build/webgpu/
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
RUN pip install --no-cache-dir --break-system-packages /tmp/webgpu_build \
    && rm -rf /tmp/webgpu_build

COPY tests/ tests/

CMD ["pytest", "tests/", "-v", "--tb=short"]