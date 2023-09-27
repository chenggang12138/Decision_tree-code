from Compiler import sorting
from sklearn import datasets,linear_model
from Compiler import mpc_math
from Compiler.types import *
from Compiler.library import *
from Compiler import util, oram
from Compiler.test import groupsum
from itertools import accumulate
import math
global max_value 
max_value = 100000
def groupprefixsum(g,x,n):
    px=Array(n,sint)
    px[0]=x[0]
    for i in range(0,n):
        for j in range(i):
            px[i]=px[i]+x[j]-g[j]*px[i]
    return px


def mgini(g,x,y,n):
    nx=Array(n,sint)
    ny=Array(n,sint)
    u0=Array(n,sint)
    u1=Array(n,sint)
    v0=Array(n,sint)
    v1=Array(n,sint)
    p=Array(n,sint)
    q=Array(n,sint)
    for i in range(0,n):
        nx[i]=1-x[i]
        ny[i]=1-y[i]
        u0[i]=nx[i]*ny[i]
        u1[i]=nx[i]*y[i]
        v0[i]=x[i]*ny[i]
        v1[i]=x[i]*y[i]
    px0=Array(n,sint)
    px0[0]=nx[0]
    for i in range(1,n):
        px0[i]=g[i]*nx[i]+(1-g[i])*(px0[i-1]+nx[i])  
    #px0=groupprefixsum(g,nx,n)
    px1=Array(n,sint)
    px1[0]=x[0]
    for i in range(1,n):
        px1[i]=g[i]*x[i]+(1-g[i])*(px1[i-1]+x[i])    
    #px1=groupprefixsum(g,x,n)
    pu0=Array(n,sint)
    pu0[0]=u0[0]
    for i in range(1,n):
        pu0[i]=g[i]*u0[i]+(1-g[i])*(pu0[i-1]+u0[i]) 
    #pu0=groupprefixsum(g,u0,n)
    pu1=Array(n,sint)
    pu1[0]=u1[0]
    for i in range(1,n):
        pu1[i]=g[i]*u1[i]+(1-g[i])*(pu1[i-1]+u1[i])     
    #pu1=groupprefixsum(g,u1,n)
    pv0=Array(n,sint)
    pv0[0]=v0[0]
    for i in range(1,n):
        pv0[i]=g[i]*v0[i]+(1-g[i])*(pv0[i-1]+v0[i])     
    #pv0=groupprefixsum(g,v0,n)
    pv1=Array(n,sint)
    pv1[0]=v1[0]
    for i in range(1,n):
        pv1[i]=g[i]*v1[i]+(1-g[i])*(pv1[i-1]+v1[i])     
    #pv1=groupprefixsum(g,v1,n)
    for i in range(n):
        p[i]=pu0[i]*pu0[i]+pu1[i]*pu1[i]
        q[i]=pv0[i]*pv0[i]+pv1[i]*pv1[i]
        c0=px0[i].__eq__(0)
        c1=px1[i].__eq__(0)
        p[i]=(1-c0)*(1-c1)*(px1[i]*p[i]+px0[i]*q[i])+(1-c0)*c1*p[i]+(1-c1)*c0*q[i]
        q[i]=(1-c0)*(1-c1)*px1[i]*px0[i]+(1-c0)*c1*px0[i]+(1-c1)*c0*px1[i]
    for i in range(0,n-1):
        p[i]=g[i+1]*p[i]
        q[i]=g[i+1]*q[i]
    p[n-1]=p[n-1]
    q[n-1]=q[n-1]
    p,vt=groupsum(g,p,n)
    q,vt=groupsum(g,q,n)
    return p,q

def vectmax(a1,a2,b,low,high):   #选择a中最大数据的索引，返回b索引下的数据choose，验证数据res low是索引下限 high是索引上限（n-1)
   if low==high:
      return b[low],a1[low],a2[low]
   else:
      mid=(low+high)//2
      left_choose,left_min1,left_min2=vectmax(a1,a2,b,low,mid)
      right_choose,right_min1,right_min2=vectmax(a1,a2,b,mid+1,high)
      condition=(left_min1*right_min2).__gt__(right_min1*left_min2)
      res1=right_min1+condition*(left_min1-right_min1)
      res2=right_min2+condition*(left_min2-right_min2)
      choose=right_choose+condition*(left_choose-right_choose)
      return choose,res1,res2

def InputAttributeSelectionFCT(g,mx,y,m,n):
    mp,mq=Matrix(m,n,sint),Matrix(m,n,sint)
    mpt,mqt=Matrix(n,m,sint),Matrix(n,m,sint)
    A=Array(n,sint)
    stand=[i+1 for i in range(0,m)]
    sstand=Array(m,sint)
    sstand.assign(stand)
    for i in range(m):
        mp[i],mq[i]=mgini(g,mx[i],y,n)
    mpt=mp.transpose()
    mqt=mq.transpose()
    for i in range(n):
        A[i],r1,r2=vectmax(mpt[i],mqt[i],sstand,0,m-1)
    #print_ln("A:%s",A.reveal())
    return A

def formatlayer(g,u,w,n):
    ng=g.same_shape()
    uu=u.same_shape()
    for i in range(0,n-1):
        ng[i]=g[i+1]
        uu[i]=g[i+1]*u[i]
    ng[n-1]=1
    uu[n-1]=u[n-1]
    sorting.radix_sort(ng, uu, n_bits=None, signed=True)    
    z=Array(w,sint)
    for i in range(0,w):
        z[i]=uu[n-w+i]
    return z   
        
def TrainInternalNodesCT(k,g,N,mx,y,m,n):
    b=Array(n,sint)
    A=Array(n,sint)
    A=InputAttributeSelectionFCT(g,mx,y,m,n) 
    AI=formatlayer(g,A,pow(2,k),n) 
    NI=formatlayer(g,N,pow(2,k),n)
    for i in range(0,n):
        for j in range(0,m):
            condition=A[i].__eq__(j+1)*mx[j][i]
            b[i]=b[i]+condition       
    #print_ln("AI:%s",AI.reveal())
    #print_ln("b:%s",b.reveal())
    return AI,NI,b

def TrainLeafNodesCT(d,g,N,y,n):
    ny=Array(n,sint)
    L=Array(n,sint)
    for i in range(n):
        ny[i]=1-y[i]
    sy,r1=groupsum(g,y,n)
    sny,r1=groupsum(g,ny,n)
    for i in range(n):
        L[i]=sny[i].__lt__(sy[i])
    NI=formatlayer(g,N,pow(2,d),n)
    LI=formatlayer(g,L,pow(2,d),n)
    #print_ln("LI:%s",LI.reveal())
    return LI,NI

def groupfirstone(g,b,n):
    bn=Array(n,sint)
    apb=Array(n,sint)
    apnb=Array(n,sint)
    pb1=Array(n,sint)
    for i in range(n):
        bn[i]=1-b[i]
    pb=groupprefixsum(g,b,n)
    pnb=groupprefixsum(g,bn,n)
    for i in range(n):
        apb[i]=pb[i]*b[i]
        apnb[i]=pnb[i]*bn[i]
    for i in range(0,n):
        pb1.__setitem__(i,apb[i].__eq__(1)+apnb[i].__eq__(1))
    return pb1

def Classification_Tree_Training(d,mx,y,m,n):
    g=Array(n,sint)
    nod=Array(n,sint)
    g[0]=1
    nod.assign_all(1)
    for i in range(1):
        AI,NI,b=TrainInternalNodesCT(i,g,nod,mx,y,m,n)
        #print_ln("AI:%s,NI:%s,b:%s",AI.reveal(),NI.reveal(),b.reveal())
        for j in range(n):
            nod[j]=pow(2,i)*b[j]+nod[j]
        pb1=groupfirstone(g,b,n)
        g.assign(pb1)
        print_ln("g:%s",g.reveal())
        sorting.radix_sort(b, g, n_bits=None, signed=True)
        sorting.radix_sort(b, nod, n_bits=None, signed=True)
        sorting.radix_sort(b, y, n_bits=None, signed=True)
        for k in range(0,m):
            sorting.radix_sort(b, mx[k], n_bits=None, signed=True)
    LI,NI=TrainLeafNodesCT(d,g,nod,y,n)
    #print_ln("LI:%s,NI:%s",LI.reveal(),NI.reveal())


def rf(loop,d,mx,y,m,n):
    for i in range(loop):
        Classification_Tree_Training(d,mx,y,m,n)





















