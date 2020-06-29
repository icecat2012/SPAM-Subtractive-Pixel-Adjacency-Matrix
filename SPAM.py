import numpy as np
from PIL import Image

def GetM3(L,C,R,T):
    # marginalization into borders
    L = np.clip(L, -T, T).flatten('F')
    C = np.clip(C, -T, T).flatten('F')
    R = np.clip(R, -T, T).flatten('F')

    # get cooccurences [-T...T]
    M = np.zeros((2*T+1,2*T+1,2*T+1))
    for i in range(-T, T+1, 1):
        C2 = C[L==i];
        R2 = R[L==i];
        for j in range(-T, T+1, 1):
            R3 = R2[C2==j];
            for k in range(-T, T+1, 1):
                M[i+T,j+T,k+T] = np.sum(R3==k)

    # normalization
    M = M.flatten('F')
    M /= np.sum(M)
    return M

def spam_extract_2(X, T):
    # horizontal left-right
    X = np.concatenate((X[:,:,0], X[:,:,1], X[:,:,2]), axis=1)
    D = X[:,:-1] - X[:,1:]
    L = D[:,2:]
    C = D[:,1:-1]
    R = D[:,:-2]
    Mh1 = GetM3(L.copy(),C.copy(),R.copy(),T)

    # horizontal right-left
    D = -D;
    L = D[:,:-2]
    C = D[:,1:-1]
    R = D[:,2:]
    Mh2 = GetM3(L.copy(),C.copy(),R.copy(),T)

    # vertical bottom top
    D = X[:-1,:] - X[1:,:]
    L = D[2:,:]
    C = D[1:-1,:]
    R = D[:-2,:]
    Mv1 = GetM3(L.copy(),C.copy(),R.copy(),T)

    # vertical top bottom
    D = -D
    L = D[:-2,:]
    C = D[1:-1,:]
    R = D[2:,:]
    Mv2 = GetM3(L.copy(),C.copy(),R.copy(),T)

    # diagonal left-right
    D = X[:-1,:-1] - X[1:,1:]
    L = D[2:,2:]
    C = D[1:-1,1:-1]
    R = D[:-2,:-2]
    Md1 = GetM3(L.copy(),C.copy(),R.copy(),T)

    # diagonal right-left
    D = -D
    L = D[:-2,:-2]
    C = D[1:-1,1:-1]
    R = D[2:,2:]
    Md2 = GetM3(L.copy(),C.copy(),R.copy(),T)

    # minor diagonal left-right
    D = X[1:,:-1] - X[:-1,1:]
    L = D[:-2,2:]
    C = D[1:-1,1:-1]
    R = D[2:,:-2]
    Mm1 = GetM3(L.copy(),C.copy(),R.copy(),T)

    # minor diagonal right-left
    D = -D
    L = D[2:,:-2]
    C = D[1:-1,1:-1]
    R = D[:-2,2:]
    Mm2 = GetM3(L.copy(),C.copy(),R.copy(),T);

    F1 = (Mh1+Mh2+Mv1+Mv2)/4;
    F2 = (Md1+Md2+Mm1+Mm2)/4;
    F = np.concatenate((F1, F2), axis=0)
    return F

def loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def SPAM(path):
     return spam_extract_2(np.array(loader(path), dtype='float'),3)

def test():
    out = SPAM('the/path/to/image.png')
    print(out)

if __name__ == '__main__':
    test()
