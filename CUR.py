import numpy as np

def get_CUR(A,k):
    Prob_c=np.square(A).sum(axis=0)/np.square(A).sum()
    Prob_r=np.square(A).sum(axis=1)/np.square(A).sum()
    def Column_Select(A,c,flag=True):
        nonlocal Prob_c,Prob_r
        if flag:
            C=np.zeros((A.shape[0],c))
            pos_array=np.zeros(c,dtype=np.int64)
            for i in range(c):
                j=np.random.choice(np.arange(A.shape[1]),p=Prob_c)
                pos_array[i]=j
                C[:,i]=A[:,j]/np.sqrt(c*Prob_c[j])
        else:
            C=np.zeros((c,A.shape[1]))
            pos_array=np.zeros(c,dtype=np.int64)
            for i in range(c):
                j=np.random.choice(np.arange(A.shape[0]),p=Prob_r)
                pos_array[i]=j
                C[i,:]=A[j,:]/np.sqrt(c*Prob_r[j])
        return C,pos_array
    C,col_pos=Column_Select(A,min(4*k,A.shape[1]),True)     # Heuri
    R,row_pos=Column_Select(A,min(4*k,A.shape[0]),False)    # Heuri
    ixgrid=np.ix_(row_pos,col_pos)
    W=A[ixgrid]
    U=np.linalg.pinv(W)
    return C,U,R
