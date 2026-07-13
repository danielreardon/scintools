"""
Shared pytest configuration for the scintools test suite.
"""

import matplotlib
matplotlib.use('Agg')  # noqa: E402 - must happen before any pyplot import
