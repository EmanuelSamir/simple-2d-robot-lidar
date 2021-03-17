import numpy as np

def sgn(x):
    if x < 0:
        return -1
    else:
        return 1


def obtain_intersection_points(x1, y1, x2, y2, xc, yc, rc):
    # https://mathworld.wolfram.com/Circle-LineIntersection.html
    x1n = x1 - xc
    x2n = x2 - xc

    y1n = y1 - yc
    y2n = y2 - yc

    dx = x2n - x1n
    dy = y2n - y1n

    dr = np.sqrt(dx**2 + dy ** 2)
    D = x1n * y2n - x2n * y1n

    delta = rc ** 2 * dr **2 - D ** 2

    if delta >= 0:
        xi1 =  (D*dy + sgn(dy) * dx  * np.sqrt( delta )) / (dr ** 2) 
        yi1 =  - ( D * dx - np.abs(dy) *  np.sqrt( delta ) ) / (dr ** 2) 

        xi2 =  (D*dy - sgn(dy) * dx  * np.sqrt( delta )) / (dr ** 2) 
        yi2 =  - ( D * dx + np.abs(dy) *  np.sqrt( delta ) ) / (dr ** 2)

        dis1 = np.sqrt( ( (xi1 + xc - x1) **2 ) + ( ( yi1 + yc - y1)**2) )
        dis2 = np.sqrt( ( (xi2 + xc - x1) **2 ) + ( ( yi2 + yc - y1)**2) )

        if dis2 > dis1:
            return True, [xi1 + xc, yi1 + yc]
        else:
            return True, [xi2 + xc, yi2 + yc]
    else: 
        return False, -1 

def validate_point(x, y, xo, yo, th, max_range):
    th = np.deg2rad(th)
    # Check if it is the same direction
    if sgn(y) != sgn(np.sin(th)):
        return False
    elif sgn(x) != sgn(np.cos(th)):
        return False
    # Check it is within the range
    elif np.sqrt(x**2 + y**2) > max_range :
        return False
    
    elif np.sqrt(x**2 + y**2) > np.sqrt(xo**2 + yo**2):
        return False
    else:
        return True