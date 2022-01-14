#!/usr/bin/env python

# This file is part of me-types-mapper.
#
#
# Copyright © 2021 Blue Brain Project/EPFL
#
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the APACHE-2 License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.apache.org/licenses/LICENSE-2.0>.

import importlib
import sys

from setuptools import setup, find_packages

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported")

# read the contents of the README file
with open("README.md", encoding="utf-8") as f:
    README = f.read()

spec = importlib.util.spec_from_file_location("me_types_mapper.version", "me_types_mapper/version.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.VERSION

with open('requirements.txt') as f:
    requirements = f.readlines()

setup(
    name="me_types_mapper",
    author="Blue Brain Project, EPFL",
    version=VERSION,
    description="Cross species inhibitory cell types mapping",
    long_description=README,
    long_description_content_type="text/x-rst",
    # url="https://bbpteam.epfl.ch/documentation/projects/cross-species-mapping",
    project_urls={
        "Tracker": "unknown",
        "Source": "https://bbpgitlab.epfl.ch/molsys/me_types_mapper",
    },
    license="BBP-internal-confidential",
    install_requires=requirements,
    packages=find_packages(),
    python_requires=">=3.6",
    extras_require={"docs": ["sphinx", "sphinx-bluebrain-theme"]},
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
