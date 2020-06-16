import numpy as np
import torch
from torch.nn import functional as F


def generateKernel(connectivity=8):
    """Generate kernel for 4 or 8 connectivity (von Neumann vs Moore neighborhood)
    
    Arguments:
        connectivity -- neighborhood type
        
    Return:
        3x3 kernel for given type of neighborhood
    """
    if connectivity == 8:
        return torch.ones((3, 3), dtype=torch.uint8)
    elif connectivity == 4:
        kernel = torch.zeros((3, 3), dtype=torch.uint8)
        kernel[1, :] = 1
        kernel[:, 1] = 1
        return kernel
    else:
        raise ValueError("Connectivity can be only 4 or 8")

def checkEdge(dots, size):
    """Check if dots are in field  

    Arguments:
        dots  -- 2-d tensor with dots in (n, 2) format (n - number of dots)
        shape -- shape of canvas
    
    Return:
        1-d tensor with single number representations
    """
    return ((dots > -1) & (dots < size)).all(dim=1)

def tile(arr, n): # todo change style
    """Numpy.tile for pytorch.

    Arguments:
        arr -- 1-d tensor
        n -- number of copies
    
    Return:
        1-d tensor with n copies of it
    """
    return torch.cat(n * [arr])

def repeat(arr, n): # todo change style
    """Numpy.repeat for pytorch.

    Arguments:
        arr -- 1-d tensor
        n -- number of copies
    
    Return:
        1-d tensor with n copies of each element
    """
    return arr.repeat(n, 1).T.reshape(-1)

def cartProduct(x, y):
    """Cartesian product for pytorch.

    Arguments:
        x -- 1-d tensor with shape (n, )
        y -- 1-d tensor with shape (m, )
    
    Return:
        2-d tensor with shape (n * m, 2)
    """
    l = tile(x, y.shape[0])
    r = repeat(y, x.shape[0])
    return torch.stack([l, r], axis=0).T

# https://discuss.pytorch.org/t/how-to-do-the-tf-gather-nd-in-pytorch/6445/12
def gather_nd(params, indices): 
    """Tensorflow.gather_nd for pytorch."""
    indices = indices.t().long()
    ndim = indices.size(0)
    idx = torch.zeros_like(indices[0]).long()
    m = 1
    
    for i in range(ndim)[::-1]:
        idx += indices[i] * m 
        m *= params.size(i)
    
    return torch.take(params, idx)

def dilation(field, kernel):
    """Dilation on field with kernel

    Arguments:
        field  -- 2-d tensor with shape (n, m)
        kernel -- 2-d tensor with shape (3, 3)
    
    Return:
        2-d tensor with shape (n, m)
    """
    kernel = kernel.view(1, 1, 3, 3).type(torch.float32)
    field = field.view(1, 1, *field.shape).type(torch.float32)
    convol = F.conv2d(field, kernel, padding=1)
    convol = torch.where(convol > 0, torch.ones_like(convol), torch.zeros_like(convol)).type(torch.uint8)
    return convol[0, 0] 

def scatter_2d_fixed_val(indecies, val, shape, device):  #, indexes, values): # todo docs
    """Analog of tensorflow.scatter_nd for 2d case and with fixed value"""
    vals = torch.ones(len(indecies)).type(torch.int64).to(device=device) * val
    return torch.sparse.IntTensor(indecies.T, vals, shape).to_dense()

def gather_2d(params, indices): # todo docs
    '''
    the input indices must be a 2d tensor in the form of [[a,b,..,c],...], 
    which represents the location of the elements.
    '''
    
    indices = indices.t().long()
    ndim = indices.size(0)
    idx = torch.zeros_like(indices[0]).long()
    m = 1
    
    for i in range(ndim)[::-1]:
        idx += indices[i] * m 
        m *= params.size(i)
    
    return torch.take(params, idx)

def dotsToNums(dots, shape):
    """Convert 2-d tensor of dots to single number representation 

    Arguments:
        dots  -- 2-d tensor with dots in (n, 2) format (n - number of dots)
        shape -- shape of canvas
    
    Return:
        1-d tensor with single number representations
    """
    return shape[0] * dots[:, 0] + dots[:, 1] 

def numsToDots(nums, shape):
    """Convert 1-d tensor with single number representations to 2-d tensor of dots 

    Arguments:
        nums  -- 1-d tensor with single number representations
        shape -- shape of canvas
    
    Return:
        2-d tensor with dots in (n, 2) format (n - number of dots)
    """
    return torch.stack((nums // shape[0], nums % shape[0]), axis=1) 

def getMoveLoc(mfield, dot, disps, device):
    """Get list of unoccupied dots near given dot in moving field

    Arguments:
        mfield -- moving field
        dot    -- 2-d vector with coords of a dot
        disps  -- search space (default: tensor [-1, 0, 1])
        device -- device we are computing on
    
    Return:
        List of unoccupied dots nearby
    """
    neighbors = cartProduct(disps, disps) + dot
    
    # Check if not outside field
    neighbors = neighbors[checkEdge(neighbors, mfield.size()[0])]
    
    # Check if not on existing dot
    nonfilled_neighbors_bool = gather_nd(mfield, neighbors) != 1
    nonfilled_neighbors = neighbors[nonfilled_neighbors_bool]
    
    # Add self position
    self_dot = torch.empty((1, 2)).to(device=device).long()
    self_dot[0] = dot
    nonfilled_neighbors = torch.cat((nonfilled_neighbors, self_dot), 0)
    return nonfilled_neighbors

def moveDots(mfield, disps, device):
    """Move all dots to nearby unoccupied dots nearby

    Arguments:
        mfield -- moving field
        disps  -- search space (default: tensor [-1, 0, 1])
        device -- device we are computing on
    
    Return:
        Updated moving field
    """
    # Get all dots list in mfield as dots:
    dots = torch.nonzero(mfield, as_tuple=False)
    for dot in dots:
        locs = getMoveLoc(mfield, dot, disps, device)
        
        # Choose only one unoccupied dot
        loc = locs[torch.randint(0, locs.shape[0] - 1, [1])[0]]

        # Move it
        mfield[dot[0], dot[1]] = 0
        mfield[loc[0], loc[1]] = 1
    return mfield

def freezeDots(ffield, mfield, kernel):
    # Make dilation with 3x3 filter
    dil = dilation(ffield, kernel)
    
    # Add mfield * dil to ffield
    ffield = ffield | (mfield & dil)
    
    # Remove dilation from mfield
    mfield = mfield & (1 - dil)
    return ffield, mfield

    
def checkUnique(dots, shape):
    """Check if dots in list are unique and return 1-d bool tensor

    Arguments:
        dots  -- 2-d tensor with dots in (n, 2) format (n - number of dots)
        shape -- shape of canvas
    
    Return:
        1-d boolean tensor with true values corresponding to unique values
    """
    dots_vals = dotsToNums(dots, shape)
    output, inverse_indices = torch.unique(dots_vals, return_inverse=False, return_counts=True)
    
    # There always more unique that non unique so it's faster to stack non unique dots
    nonUnique = output[inverse_indices > 1]
    nonUnique_bool = [dots_vals == nonUnique[i] for i in range(len(nonUnique))]
    if len(nonUnique_bool) > 0:
        nonUnique_bool = torch.stack(nonUnique_bool, axis = 0)
        nonUnique_bool = torch.max(nonUnique_bool, axis = 0)
        return ~nonUnique_bool.values  # invert output
    else:
        return torch.zeros_like(dots_vals) == 0
    

def newMoveDots(mfield, device): # todo docs
    # Get all dots in mfield as dots (n, 2):
    dots = torch.nonzero(mfield, as_tuple=False)

    # Make pairs of coords (before and after movement)
    dots = dots.repeat(1, 2)
    
    # Add random perturbation (+- 1 on both axes)
    rands = torch.zeros((dots.shape[0], 2), dtype=torch.int64, device=device)
    rands = torch.cat((rands, torch.randint(-1, 2, (dots.shape[0], 2), device=device)), axis=1)
    newDots = dots + rands

    # Check edges
    checkedEdges = checkEdge(newDots[:, 2:], mfield.shape[0]).view(-1, 1).repeat(1, 4)
    newDots = torch.where(checkedEdges, newDots, dots)

    # Check unique
    # Repeat nonunique removal until none found
    while True:
        checkUnique_bool = checkUnique(newDots[:, 2:], mfield.shape).view(-1, 1).repeat(1, 4)
        if len(checkUnique_bool) == 0 or torch.min(checkUnique_bool).cpu().numpy():
            break
        newDots = torch.where(checkUnique_bool, newDots, dots)
    newField = scatter_2d_fixed_val(newDots[:, 2:], 1, mfield.shape, device)
    return newField