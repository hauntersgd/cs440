import numpy as np

def estimate_geometric(PX):
    '''
    @param:
    PX (numpy array of length cX): PX[x] = P(X=x), the observed probability mass function

    @return:
    p (scalar): the parameter of a matching geometric random variable
    PY (numpy array of length cX): PY[x] = P(Y=y), the first cX values of the pmf of a
      geometric random variable such that E[Y]=E[X].
    '''
    #raise RuntimeError("You need to write this")
    Ex = 0
    PY = np.zeros(PX.size)

    for x, pmf in enumerate(PX):
        Ex += x * pmf
    
    p = 1 / (1 + Ex)

    for y in range(PY.size):
        PY[y] = p*((1-p)**y)

    return p, PY
