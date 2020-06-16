import numpy as np
from skimage import morphology as mpl
import matplotlib.pylab as plt

def printFields(fields, title=None):
    """Ascii-style visualisation of fields
    
    Arguments:
        fields -- list of 2-d arrays
        title  -- name diplayed above fields
    """
    fields = np.array(fields)
    minuses = ((fields.shape[2] + 1) * fields.shape[0] ) - 1
    seperator = "+" + "-" * minuses + "+"
    if title:
        minuses2 = ((fields.shape[2] + 1) * fields.shape[0] ) - 3 - len(title)
        seperator2 = "+" + "-" * (minuses2 // 2) + "<" + title + ">" + "-" * (minuses2 // 2 + minuses2 % 2) + "+"
        print(seperator2)
    else:
        print(seperator)
    for i in range(fields.shape[1]):
        print("|", end='')
        for j in range(fields.shape[0]):
            for k in range(fields.shape[2]):
                print("*" if fields[j, i, k] == 1 else '.', end='')
            print("|", end='')
        print("")
    print(seperator)

def placeDots(field, particlesNum):
    """Generate fixed number of particles on a field
    
    Arguments:
        field        -- 2-d arrays to fill
        particlesNum -- number of particles
    """
    gridSize = field.shape[0]
    positions = np.arange(gridSize * gridSize)
    np.random.shuffle(positions)
    positions = positions[:particlesNum]
    # Retreaving coords from values
    ypos = positions // gridSize
    xpos = positions % gridSize
    field[xpos, ypos] = 1

def freezeDots(ffield, mfield):
    """Move dots from moving field to frozen field if found nearby (moore neighborhood)
    
    Arguments:
        ffield -- frozen field
        mfield -- moving field
        
    Return:
        Updated frozen field and moving field
    """
    # Make dilation with 3x3 filter
    dil = mpl.binary_dilation(ffield, np.ones((3, 3))).astype(np.uint8)
    # Add mfield * dil to ffield
    ffield = ffield | (mfield & dil)
    # Remove dilation from mfield
    mfield = mfield & (1 - dil)
    return ffield, mfield

def checkEdge(size, numj, numi):
    """Check if dot is in field  

    Arguments:
        size  -- field size
        numj -- position on y axis
        numi -- position on x axis
    
    Return:
        True if dot is in field
    """
    return (numi > -1) and (numj > -1) and (numi < size) and (numj < size)


def getMoveLoc(mfield, dot):
    """Get list of unoccupied dots near given dot in moving field

    Arguments:
        mfield -- moving field
        dot    -- 2-d vector with coords of a dot
    
    Return:
        List of unoccupied dots nearby (moore neighborhood)
    """
    n = mfield.shape[0]
    neighbors = [dot]
    for j in range(-1, 2):
        for i in range(-1, 2):
            # Check if not outside field
            if checkEdge(n, dot[0] + j, dot[1] + i):
                # Check if not on existing dot
                if mfield[dot[0] + j, dot[1] + i] != 1:
                    neighbors.append((dot[0] + j, dot[1] + i))
    return neighbors

def moveDots(mfield):
    """Move all dots to nearby unoccupied dots nearby (moore neighborhood)

    Arguments:
        mfield -- moving field
    
    Return:
        Updated moving field
    """
    # Get list of all dots in mfield as dots:
    dots = np.argwhere(mfield == 1)

    # Shuffle l (to remove any bias due to order of dots)
    np.random.shuffle(dots)
    
    for dot in dots:
        loc = getMoveLoc(mfield, dot)
        
        # Choose only one unoccupied dot
        loc = loc[np.random.choice(np.arange(len(loc)))]

        # Move it
        mfield[dot[0], dot[1]] = 0
        mfield[loc[0], loc[1]] = 1
    return mfield
    