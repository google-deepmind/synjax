#!/bin/bash
# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Runs CI tests on a local machine.
set -xeuo pipefail

# Install deps in a virtual env.
readonly VENV_DIR=/tmp/synjax-env
rm -rf "${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python --version

# Install dependencies.
pip install --upgrade pip setuptools wheel
pip install flake8 pytest-xdist pytest-forked pytype pylint pylint-exit
pip install -r requirements-test.txt
pip install -r requirements.txt

# Lint with flake8.
flake8 `find synjax -name '*.py' | xargs` --count --select=E9,F63,F70,E225,E251 --show-source --statistics

# Lint with pylint.
# Fail on errors, warning, conventions and refactoring messages.
PYLINT_ARGS="-efail -wfail -cfail -rfail"
# Download Google OSS config.
wget -nd -v -t 3 -O .pylintrc https://google.github.io/styleguide/pylintrc
# Append specific config lines.
echo "disable=abstract-method,unnecessary-lambda-assignment,no-value-for-parameter,use-dict-literal" >> .pylintrc
# Lint modules and tests separately.
# Disable `abstract-method` warnings.
pylint --rcfile=.pylintrc `find synjax -name '*.py' | grep -v 'test.py' | xargs` || pylint-exit $PYLINT_ARGS $?
# Disable `protected-access`, `arguments-differ`, `not-callable`, `invalid-unary-operand-type` warnings and errors for tests.
pylint --rcfile=.pylintrc `find synjax -name '*_test.py' | xargs` -d W0212,W0221,E1102,E1130 || pylint-exit $PYLINT_ARGS $?
# Cleanup.
rm .pylintrc

# Build the package.
python setup.py sdist
pip wheel --verbose --no-deps --no-clean dist/synjax*.tar.gz
pip install synjax*.whl

# Check types with pytype.
pytype `find synjax/_src/ -name "*py" | xargs` -k

# Run tests using pytest.
# Change directory to avoid importing the package from repo root.
mkdir _testing && cd _testing

# Main tests.

# Disable JAX optimizations to speed up tests.
export JAX_DISABLE_MOST_OPTIMIZATIONS=True
# Disables jaxtyping runtime checks while running this test script.
export PYTHONOPTIMIZE=1
pytest -n"$(grep -c ^processor /proc/cpuinfo)" --forked `find ../synjax/_src/ -name "*_test.py" | grep -v "/distribution_test.py" | sort`
unset JAX_DISABLE_MOST_OPTIMIZATIONS
unset PYTHONOPTIMIZE

cd ..

set +u
deactivate
echo "All tests passed. Congrats!"
