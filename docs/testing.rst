Testing framework
=================

The ``webgpu.testing`` module provides reusable pytest infrastructure for
visual regression tests that run against a real WebGPU device inside headless
Chrome.  It is designed so that downstream packages (such as
`ngsolve_webgpu <https://github.com/CERBSim/ngsolve_webgpu>`_) can set up
their own test suites with minimal boilerplate.


Architecture
------------

Tests run inside a Docker container that provides:

* **Headless Chrome** with WebGPU enabled
* **Lavapipe** (Mesa software Vulkan) so no physical GPU is required
* **Playwright** to drive the browser from Python

The ``webgpu.testing`` module then:

1. Launches a websocket server (``webgpu.platform.init``)
2. Serves a minimal HTML page that connects back to Python
3. Provides pytest fixtures that give tests access to the live WebGPU
   device, a browser page and helper methods for screenshots and baseline
   comparison


Docker images
-------------

A three-layer Docker image scheme keeps things DRY:

``Dockerfile.base`` (provided by webgpu)
   Playwright + Chrome + lavapipe + the ``webgpu`` package installed.
   Clean ``/app`` work directory, no tests.  Downstream packages derive
   from this image.

``Dockerfile`` (webgpu tests)
   Extends the base image, copies the webgpu test suite into ``/app/tests``.

Downstream ``Dockerfile`` (e.g. ngsolve_webgpu)
   Extends the base image, installs additional dependencies and the
   downstream package, copies its own tests.

.. code-block:: text

   ┌──────────────────────────────┐
   │  Dockerfile.base             │
   │  (playwright, chrome,        │
   │   lavapipe, webgpu)          │
   ├──────────────┬───────────────┤
   │              │               │
   │  Dockerfile  │  downstream   │
   │  (webgpu     │  Dockerfile   │
   │   tests)     │  (+ ngsolve,  │
   │              │   own tests)  │
   └──────────────┴───────────────┘


Container registry
^^^^^^^^^^^^^^^^^^

The webgpu CI publishes the base image to the GitHub Container Registry on
every push to ``main``:

.. code-block:: text

   ghcr.io/cerbsim/webgpu-base:latest

Downstream packages can pull this pre-built image instead of rebuilding it
from source, which saves several minutes of CI time.

**Building locally** (from a webgpu checkout)::

   docker build -f tests/Dockerfile.base -t webgpu-base .

**Pulling from the registry**::

   docker pull ghcr.io/cerbsim/webgpu-base:latest

**Using in a downstream Dockerfile**:

.. code-block:: dockerfile

   ARG BASE_IMAGE=ghcr.io/cerbsim/webgpu-base:latest
   FROM ${BASE_IMAGE}

   # install your package ...


Provided fixtures
-----------------

Register the fixtures by adding a single line to your ``conftest.py``:

.. code-block:: python

   pytest_plugins = ["webgpu.testing"]

The following fixtures then become available:

``browser`` *(session-scoped)*
   A Playwright ``Browser`` instance (headless Chrome with WebGPU flags).

``page`` *(function-scoped)*
   A fresh browser page with no webgpu connection.  Useful for pure-JS
   smoke tests (e.g. checking that ``navigator.gpu`` exists).

``webgpu_env`` *(session-scoped)*
   A fully initialised :class:`~webgpu.testing.WebGPUTestEnv` with a live
   websocket bridge between Python and the browser.  This is the main
   fixture for rendering tests.


WebGPUTestEnv
-------------

The ``webgpu_env`` fixture yields a
:class:`~webgpu.testing.WebGPUTestEnv` instance with the following
attributes and methods:

.. attribute:: page

   The Playwright ``Page`` connected to the websocket bridge.

.. attribute:: platform

   The ``webgpu.platform`` module (gives access to ``platform.js``).

.. attribute:: wj

   The ``webgpu.jupyter`` module (patched for headless use).

.. attribute:: output_dir

   ``Path`` where test output images are written.  Must be set by the
   downstream ``conftest.py`` (see below).

.. attribute:: baseline_dir

   ``Path`` where reference images are stored.  Must be set by the
   downstream ``conftest.py``.

.. method:: ensure_canvas(width=600, height=600)

   Inject a ``<canvas>`` element into the browser page that matches the
   next ``wj.Draw()`` call.  Returns the canvas element's ``id``.

.. method:: screenshot(name, canvas_id=None)

   Take a Playwright screenshot of a canvas element and save it to
   ``output_dir / "{name}.png"``.  Returns the output ``Path``.

.. method:: readback_texture(scene, path)

   Read back the rendered texture from the GPU via a JS-side buffer
   readback and save it as a PNG. Returns *path*.

.. method:: assert_matches_baseline(scene, filename, *, threshold=0.01)

   Perform a full visual regression check on a rendered scene:

   1. Assert the scene is valid and has render objects
   2. Wait 500 ms for rendering to settle
   3. Read back the GPU texture to ``output_dir / filename``
   4. Compare the output against ``baseline_dir / filename``

   Fails the test if more than *threshold* (fraction) of pixels differ.

   When the environment variable ``UPDATE_BASELINES=1`` is set, the
   output is copied **to** the baseline instead of compared, making it
   easy to regenerate references.


Quick start for downstream packages
------------------------------------

This section walks through adding a visual test suite to a package that
builds on ``webgpu``.  The
`ngsolve_webgpu test suite <https://github.com/CERBSim/ngsolve_webgpu/tree/main/tests>`_
is a complete working example of this pattern.

1. **Create** ``tests/conftest.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   """conftest.py — register webgpu.testing fixtures and set directories."""

   from pathlib import Path
   import pytest

   pytest_plugins = ["webgpu.testing"]

   TESTS_DIR = Path(__file__).parent

   @pytest.fixture(scope="session", autouse=True)
   def _configure_dirs(webgpu_env):
       webgpu_env.output_dir = TESTS_DIR / "output"
       webgpu_env.baseline_dir = TESTS_DIR / "baselines"

This is all the setup needed.  Every fixture from ``webgpu.testing`` is
now available in your tests.

2. **Write tests**
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   """test_rendering.py"""

   class TestMyRendering:
       def test_draw_something(self, webgpu_env):
           # Import your package lazily — see note below.
           from my_package.jupyter import Draw

           webgpu_env.ensure_canvas(600, 600)
           scene = Draw(my_data, width=600, height=600)

           # Readback, validate, and compare against baseline — all in one call
           webgpu_env.assert_matches_baseline(scene, "my_test.png")

.. important::

   Packages that trigger ``webgpu.jupyter`` at import time (which calls
   ``platform.init()`` and blocks on a websocket connection) **must** be
   imported inside the test function, not at module level.  Otherwise
   pytest will hang during test collection.

3. **Create a Dockerfile**
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Derive from the ``webgpu-base`` image so you get Chrome, lavapipe, and the
``webgpu`` package (including ``webgpu.testing``) for free:

.. code-block:: dockerfile

   ARG BASE_IMAGE=ghcr.io/cerbsim/webgpu-base:latest
   FROM ${BASE_IMAGE}

   RUN pip install --no-cache-dir --break-system-packages my-dependency

   WORKDIR /app
   COPY pyproject.toml .
   COPY my_package/ my_package/
   ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
   RUN pip install --no-cache-dir --break-system-packages .

   COPY tests/ tests/

   CMD ["pytest", "tests/", "-v", "--tb=short"]

4. **Create** ``tests/run_tests.sh``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   #!/usr/bin/env bash
   set -e
   cd "$(dirname "$0")/.."

   BASE_IMAGE=webgpu-base
   IMAGE=my-package-tests

   # Build the webgpu base image from the sibling checkout
   WEBGPU_DIR="$(cd ../webgpu && pwd)"
   echo "==> Building base image..."
   docker build -f "$WEBGPU_DIR/tests/Dockerfile.base" \
       -t "$BASE_IMAGE" "$WEBGPU_DIR"

   echo "==> Building test image..."
   docker build -f tests/Dockerfile \
       --build-arg BASE_IMAGE="$BASE_IMAGE" -t "$IMAGE" .

   echo "==> Running tests..."
   docker run --rm \
       -v "$(pwd)/tests/output:/app/tests/output" \
       -v "$(pwd)/tests/baselines:/app/tests/baselines" \
       "$IMAGE"

5. **Generate initial baselines**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the tests once with ``UPDATE_BASELINES=1`` to create the reference
images::

   UPDATE_BASELINES=1 ./tests/run_tests.sh

Or pass the variable through Docker::

   docker run --rm -e UPDATE_BASELINES=1 \
       -v "$(pwd)/tests/baselines:/app/tests/baselines" \
       my-package-tests

The generated PNGs in ``tests/baselines/`` should be committed to version
control.  If your repository uses **Git LFS** for binary files (recommended
for PNGs), make sure LFS is set up before committing::

   git lfs track "*.png"
   git add .gitattributes tests/baselines/
   git commit -m "Add baseline images"


GitHub Actions
--------------

The webgpu CI publishes ``ghcr.io/cerbsim/webgpu-base:latest`` on every
push to ``main``.  Downstream packages can pull this image directly,
avoiding the need to check out the webgpu repository or rebuild the base
image.

A minimal workflow for a downstream package:

.. code-block:: yaml

   name: Tests
   on:
     push:
       branches: [main]
     pull_request:
       branches: [main]

   env:
     BASE_IMAGE: ghcr.io/cerbsim/webgpu-base

   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
           with:
             lfs: true  # needed if baselines are stored in Git LFS

         - name: Pull base image
           run: docker pull ${{ env.BASE_IMAGE }}:latest

         - name: Build test image
           run: |
             docker build -f tests/Dockerfile \
               --build-arg BASE_IMAGE=${{ env.BASE_IMAGE }}:latest \
               -t my-tests .

         - name: Run tests
           run: |
             docker run --rm \
               -v ${{ github.workspace }}/tests/output:/app/tests/output \
               my-tests

         - name: Upload output on failure
           if: failure()
           uses: actions/upload-artifact@v4
           with:
             name: test-output
             path: tests/output/

The test image build uses plain ``docker build`` (not buildx) so it can
see the pulled base image in the local Docker daemon.  The test image
layer is small (just installing your package + copying tests), so caching
it is not necessary.