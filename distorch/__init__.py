import importlib.util

__version__ = "0.0.1"

use_pykeops = importlib.util.find_spec('pykeops') is not None
