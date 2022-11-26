import numpy as np

def depth2intensity(depth, interval=300):
    """
    Function for convertion rainfall depth (in mm) to
    rainfall intensity (mm/h)
    Args:
        depth: float
        float or array of float
        rainfall depth (mm)
        interval : number
        time interval (in sec) which is correspondend to depth values
    Returns:
        intensity: float
        float or array of float
        rainfall intensity (mm/h)
    """
    # print(f"Depth: {depth}")
    return depth * 3600 / interval

def RYScaler(X_mm):
    '''
    Scale RY data from mm (in float64) to brightness (in uint8).
    Args:
        X (numpy.ndarray): RY radar image
    Returns:
        numpy.ndarray(uint8): brightness integer values from 0 to 255
                              for corresponding input rainfall intensity
        float: c1, scaling coefficient
        float: c2, scaling coefficient
    '''
    def mmh2rfl(r, a=256., b=1.42):
        '''
        .. based on wradlib.zr.r2z function
        .. r --> z
        '''
        return a * r ** b

    def rfl2dbz(z):
        '''
        .. based on wradlib.trafo.decibel function
        .. z --> d
        '''
        return 10. * np.log10(z)

    # mm to mm/h
    X_mmh = depth2intensity(X_mm)
    # mm/h to reflectivity
    X_rfl = mmh2rfl(X_mmh)
    # remove zero reflectivity
    # then log10(0.1) = -1 not inf (numpy warning arised)
    X_rfl[X_rfl == 0] = 0.1
    # reflectivity to dBz
    X_dbz = rfl2dbz(X_rfl)
    # remove all -inf
    X_dbz[X_dbz < 0] = 0

    # MinMaxScaling
    c1 = X_dbz.min()
    c2 = X_dbz.max()

    return ((X_dbz - c1) / (c2 - c1) * 255).astype(np.uint8), c1, c2
