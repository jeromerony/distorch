__version__ = "0.0.1"

use_pykeops = True
try:
    import pykeops

    pykeops.test_torch_bindings()
except:
    import warnings

    warnings.warn('PyKeops could not be imported, this will result in high memory usage and/or out-of-memory crash.')
    use_pykeops = False
