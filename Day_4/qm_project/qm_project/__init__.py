"""
qm_project
A package for doing Hartree-Fock/MP2
"""

# Add imports here
from .run import run

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
