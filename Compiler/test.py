from Compiler import sorting
from sklearn import datasets,linear_model
from Compiler import mpc_math
from Compiler.types import *
from Compiler.library import *
from Compiler import util, oram

from itertools import accumulate
import math
global max_value 
max_value = 100000
def groupprefixsum(g,x,n):
    px=Array(n,sint)
    px[0]=x[0]
    for i in range(1,n):
        px[i]=g[i]*x[i]+(1-g[i])*(px[i-1]+x[i])
    return px

def groupprefixmin(g,numerator,denominator,y,n):
    p1=Array(n,sint)
    p2=Array(n,sint)
    qx=Array(n,sint)
    p1[0]=numerator[0]
    p2[0]=denominator[0]
    qx[0]=y[0]
    @for_range(1,n)
    def _(i): 
        condition=(p1[i-1]*denominator[i]).__le__(numerator[i]*p2[i-1])
        p1[i]=g[i]*numerator[i]+(1-g[i])*(condition*p1[i-1]+(1-condition)*numerator[i])
        p2[i]=g[i]*denominator[i]+(1-g[i])*(condition*p2[i-1]+(1-condition)*denominator[i])
        qx[i]=g[i]*y[i]+(1-g[i])*(condition*qx[i-1]+(1-condition)*y[i])
    return p1,p2,qx
def groupsum(g,x,n):      #px为前缀和
    px=groupprefixsum(g,x,n)
    #print_ln("px:%s",px.reveal())
    vt=Array(n,sint)
    vz=Array(n,sint)
    vpx2=Array(n,sint)
    rt=Array(n,sint)
    rz=Array(n,sint)
    for i in range(0,n-1):
        vt[i]=g[i+1]
        vz[i]=vt[i]*px[i]
    vt[n-1]=1
    vz[n-1]=px[n-1]
    #clear_vt=vt.reveal()
    clear_vz=vz.reveal()
    #print_ln("vt:%s",vt.reveal())
    #print_ln("vz:%s",vz.reveal())
    rt=vt.get_reverse_vector()
    rz=vz.get_reverse_vector()
    #print_ln("rt:%s",rt.reveal())
    #print_ln("rz:%s",rz.reveal())
    px2=groupprefixsum(rt,rz,n)
    clear_vz=px2.reveal()
    #print_ln("px2:%s",px2.reveal())
    vpx2=px2.get_reverse_vector()
    #print_ln("vpx2:%s",vpx2.reveal())
    return vpx2,px

def groupmin(g,x1,x2,y,n):      #px为前缀和
    px1,px2,qx=groupprefixmin(g,x1,x2,y,n)
    #print_ln("px:%s",px.reveal())
    vt=Array(n,sint)
    vz=Array(n,sint)
    vz1=Array(n,sint)
    vz2=Array(n,sint)
    vpx1=Array(n,sint)
    vpx2=Array(n,sint)
    rt=Array(n,sint)
    rz=Array(n,sint)
    rz1=Array(n,sint)
    rz2=Array(n,sint)
    for i in range(0,n-1):
        vt[i]=g[i+1]
        vz[i]=vt[i]*px1[i]
        vz1[i]=vt[i]*px2[i]
        vz2[i]=vt[i]*qx[i]
    vt[n-1]=1
    vz[n-1]=px1[n-1]
    vz1[n-1]=px2[n-1]
    vz2[n-1]=qx[n-1]
    #clear_vt=vt.reveal()
    clear_vz=vz.reveal()
    #print_ln("vt:%s",vt.reveal())
    #print_ln("vz:%s",vz.reveal())
    rt=vt.get_reverse_vector()
    rz=vz.get_reverse_vector()
    rz1=vz1.get_reverse_vector()
    rz2=vz2.get_reverse_vector()
    #print_ln("rt:%s",rt.reveal())
    #print_ln("rz:%s",rz.reveal())
    ppx=groupprefixsum(rt,rz,n)
    ppx2=groupprefixsum(rt,rz1,n)
    qx2=groupprefixsum(rt,rz2,n)
    clear_vz=px2.reveal()
    #print_ln("px2:%s",px2.reveal())
    vppx=ppx.get_reverse_vector()
    vppx2=ppx2.get_reverse_vector()
    vqx2=qx2.get_reverse_vector()
    #print_ln("vpx2:%s",vpx2.reveal())
    return vppx,vppx2,vqx2

def mse(g,y,n):
    ave_y=Array(n,sint)
    y2=Array(n,sint)
    ry2=Array(n,sint)
    ry=Array(n,sint)
    rn=Array(n,sint)
    ln2=Array(n,sint)
    rn2=Array(n,sint)
    vone=Array(n,sint)
    vone.assign_all(1)
    y2=y*y
    sy2,ly2=groupsum(g,y2,n)
    sy,ly=groupsum(g,y,n)
    sn,ln=groupsum(g,vone,n)
    for i in range(0,n):
        ry2[i]=sy2[i]-ly2[i]
        ry[i]=sy[i]-ly[i]
        rn[i]=sn[i]-ln[i]
    #print_ln("ry2:%s",ry2.reveal())
    #print_ln("ry:%s",ry.reveal())
    #print_ln("rn:%s",rn.reveal())
    ln2=ln*ln
    rn2=rn*rn
    pl=Array(n,sint)
    pr=Array(n,sint)
    p=Array(n,sint)
    q=Array(n,sint)
    for i in range(0,n):
        pl[i]=ly2[i]*ln2[i]-ln[i]*ly[i]*ly[i]
        pr[i]=ry2[i]*rn2[i]-rn[i]*ry[i]*ry[i]       
    condition=Array(n,sint)
    for i in range(0,n):
        condition[i]=pr[i].__eq__(0)
        p.__setitem__(i,condition[i]*pl[i]+(1-condition[i])*(pl[i]*rn2[i]+pr[i]*ln2[i]))
        q.__setitem__(i,condition[i]*ln2[i]+(1-condition[i])*(rn2[i]*ln2[i]))
    #print_ln("pl:%s",pl.reveal())
    #print_ln("pr:%s",pr.reveal())
    #print_ln("condition:%s",condition.reveal())
    #print_ln("ln2:%s",ln2.reveal())
    #print_ln("rn2:%s",rn2.reveal())
    return p,q

def sigle_attr(g,x,y,n):
    vect_t=Array(n,sint)
    #no=Array(n,sfix)
    p,q=mse(g,y,n)
    print_ln("p:%s q:%s",p.reveal(),q.reveal())    
    for i in range(0,n-1):
        vect_t[i]=x[i]+x[i+1]+g[i+1]*(x[i]-x[i+1])
    vect_t[n-1]=2*x[n-1]
    #print_ln("%s",vect_t.reveal())
    p1,p2,q1=groupmin(g,p,q,vect_t,n)
    return p1,p2,q1

def vectmin(a1,a2,b,low,high):   #选择a中最大数据的索引，返回b索引下的数据choose，验证数据res low是索引下限 high是索引上限（n-1)
   if low==high:
      return b[low],a1[low],a2[low]
   else:
      mid=(low+high)//2
      left_choose,left_min1,left_min2=vectmin(a1,a2,b,low,mid)
      right_choose,right_min1,right_min2=vectmin(a1,a2,b,mid+1,high)
      condition=(left_min1*right_min2) .__lt__(right_min1*left_min2)
      res1=right_min1+condition*(left_min1-right_min1)
      res2=right_min2+condition*(left_min2-right_min2)
      choose=right_choose+condition*(left_choose-right_choose)
      return choose,res1,res2

def all_attr(g,mx,y,m,n):
    xi=Array(n,sint)  
    u_one=Matrix(m,n,sint) 
    u_two=Matrix(m,n,sint) 
    vm=Matrix(m,n,sint) 
    @for_range_opt(m)
    def _(i):
        sub_D=Matrix(2,n,sint)
        sub_D[0].assign(mx[i])
        sub_D[1].assign(y)
        pg=Array(n,sint)
        pg[0]=1
        for j in range(1,n):
            pg[j]=pg[j-1]+g[j]
        sorting.radix_sort(sub_D[0], pg, n_bits=None, signed=True)
        sorting.radix_sort(sub_D[0], sub_D[1], n_bits=None, signed=True)
        sorting.radix_sort(sub_D[0], sub_D[0], n_bits=None, signed=True)
        sorting.radix_sort(pg,sub_D[0], n_bits=None, signed=True)
        sorting.radix_sort(pg,sub_D[1], n_bits=None, signed=True)
        u_one[i],u_two[i],vm[i]=sigle_attr(g,sub_D[0],sub_D[1],n) 
        print_ln("%s , %s , %s ",u_one[i].reveal(),u_two[i].reveal(),vm[i].reveal())
    u_one=u_one.transpose()
    u_two=u_two.transpose()
    vm=vm.transpose()
    Aa=Array(n,sint)
    Ta=Array(n,sint)
    stand=[i+1 for i in range(0,m)]
    sstand=Array(m,sint)
    sstand.assign(stand)
    @for_range_opt(n)
    def _(i):
        Aa[i],no1,no2=vectmin(u_one[i],u_two[i],sstand,0,m-1)
        Ta[i],no3,no4=vectmin(u_one[i],u_two[i],vm[i],0,m-1)
        ##print_ln("1:%s and 2:%s",no1.reveal(),no2.reveal())
    ##print_ln("A:%s and T:%s",Aa.reveal(),Ta.reveal())
    return Aa,Ta    

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

def inner_node(k,mx,y,g,nod,n,m):
    Aa,Ta=all_attr(g,mx,y,m,n)
    print_ln("Aa:%s",Aa.reveal())
    AID=formatlayer(g,Aa,pow(2,k),n)
    Threshold=formatlayer(g,Ta,pow(2,k),n)
    NID=formatlayer(g,nod,pow(2,k),n)
    xk=Array(n,sint)
    print_ln("Aa:%s",Aa.reveal())
    for i in range(0,n):
        for j in range(0,m):
            condition=(Aa[i].__eq__(j+1))*mx[j][i]
            xk[i]=xk[i]+condition
    print_ln("xk:%s",xk.reveal())
    b=Array(n,sint)
    for i in range(0,n):
        b[i]=(xk[i]*2).__gt__(Ta[i])
    return AID,Threshold,NID,b

#AID,Threshold,NID,b=inner_node(2,x,y,g,nod,n,2)
#print_ln("AID:%s",AID.reveal()) 
#print_ln("Threshold:%s",Threshold.reveal())
#print_ln("b:%s",b.reveal())


def leaf_node(h,g,y,nod,n):
    pnq=Matrix(2,n,sint) 
    pnq[0],no1=groupsum(g,y,n) 
    one=Array(n,sint)
    one.assign_all(1)
    pnq[1],no2=groupsum(g,one,n)
    NID=formatlayer(g,nod,pow(2,h),n)
    LableP=formatlayer(g,pnq[0],pow(2,h),n)
    LableQ=formatlayer(g,pnq[1],pow(2,h),n)
    return NID,LableP,LableQ 
#print_ln("y:%s",y.reveal())
#NID,LableP,LableQ=leaf_node(2,g,y,nod,n)
#print_ln("NID:%s",NID.reveal()) 
#print_ln("LableP:%s,LableQ:%s",LableP.reveal(),LableQ.reveal())

def groupfirstone(g,b,n):
    pb=groupprefixsum(g,b,n)
    pb=pb*b
    pb1=Array(n,sint)
    for i in range(0,n):
        pb1.__setitem__(i,pb[i].__eq__(1))
    return pb1

def regression_tree(mx,y,h,m,n):
    NID=Matrix(h,n,sint)
    AID=Matrix(h,n,sint)
    Threshold=Matrix(h,n,sint)
    nod=Array(n,sint)
    nod.assign_all(1)
    g=Array(n,sint)
    g.assign_all(0)
    g[0]=1
    AIDM,ThresholdM,NIDM=Matrix(h,n,sint),Matrix(h,n,sint),Matrix(h,n,sint)
    for i in range(0,h):
        print_ln("y:%s",y.reveal()) 
        print_ln("g:%s",g.reveal())  
        print_ln("nod:%s",nod.reveal())         
        AIDM[i],ThresholdM[i],NIDM[i],b=inner_node(i,mx,y,g,nod,n,m) 
        print_ln("A:%s and T:%s nid:%s b:%s",AIDM[i].reveal(),ThresholdM[i].reveal(),NIDM[i].reveal(),b.reveal())
        nb=Array(n,sint)
        for j in range(0,n):
            nod[j]=pow(2,i)*b[j]+nod[j]
            nb[j]=1-b[j]
        pb1=groupfirstone(g,b,n)
        pb2=groupfirstone(g,nb,n)
        for j in range(0,n):
            g[j]=pb1[j]+pb2[j]   
        sorting.radix_sort(b, g, n_bits=None, signed=True)
        sorting.radix_sort(b, nod, n_bits=None, signed=True)
        sorting.radix_sort(b, y, n_bits=None, signed=True)
        for i in range(0,m):
            sorting.radix_sort(b, mx[i], n_bits=None, signed=True)
    LNIDM,LableP,LableQ=leaf_node(h,g,y,nod,n)
    print_ln("LNIDM:%s,LableP:%s,LableQ:%s",LNIDM.reveal(),LableP.reveal(),LableQ.reveal())  
    return  AIDM,ThresholdM,NIDM,LNIDM,LableP,LableQ

def decision_predict(AIDM,ThresholdM,LableP,LableQ,mx,h,m,n):
    chnod=Array(n,sint)
    chnod.assign_all(0)
    cm=Array(n,sint)
    ct=Array(n,sint)
    ca=Array(n,sint)
    mx=mx.transpose()
    b=Array(n,sint)
    conditon=sint(0)
    for i in range(0,h):
        cm.assign_all(0)
        ct.assign_all(0)
        b.assign_all(0)
        ca.assign_all(0)
        for j in range(0,n):
            for k2 in range(0,pow(2,i)):
                condition=chnod[j].__eq__(k2)
                ct[j]=ct[j]+condition*ThresholdM[i][k2]
                ca[j]=ca[j]+condition*AIDM[i][k2]
            for k in range(0,m):
                cm[j]=cm[j]+((ca[j]-1).__eq__(k))*mx[j][k]            
            b[j]=(cm[j]*2).__gt__(ct[j])
            chnod[j]=chnod[j]+b[j]*pow(2,i)
        print_ln("cm:%s",cm.reveal())
        print_ln("ct:%s",ct.reveal())
        print_ln("ca:%s",ca.reveal())
        print_ln("chnod:%s",chnod.reveal())
    re1=Array(n,sint)
    re1.assign_all(0)
    re2=Array(n,sint)
    re2.assign_all(0)
    for i in range(0,n):
        for j in range(0,pow(2,h)):
            re1[i]=re1[i]+(chnod[i].__eq__(j))*LableP[j]
            re2[i]=re2[i]+(chnod[i].__eq__(j))*LableQ[j]
    return re1,re2  
def gbrt(mx,y,loop,m,n,depth):
    sy=sint(0)
    y1=Array(n,sint) 
    for i in range(0,n):
        y1[i]=y[i]
    fy=Array(n,sint) 
    for i in range(0,n):
        sy=sy+y[i]
    for i in range(0,n):
        fy[i]=sint(sfix(sy)/n)
    print_ln("fy:%s",fy.reveal())
    for i in range(0,loop):
        for j in range(0,n):
            y[j]=y[j]-fy[j] 
        AIDM1,ThresholdM1,NIDM1,LNIDM1,LableP1,LableQ1=regression_tree(mx,y,depth,m,n)
        r1,r2=decision_predict(AIDM1,ThresholdM1,LableP1,LableQ1,mx,depth,m,n)
        for j in range(0,n):
            fy[j]=sint(sfix(r1[j])/sfix(r2[j]))
    mse=sint(0)
    for i in range(0,n):
        mse=mse+(y1[i]-y[i])*(y1[i]-y[i])
    print_ln("mse:%s",mse.reveal())

def rft(loop,d,mx,y,m,n):
    @for_range_opt(n)
    def _(i):
        regression_tree(mx,y,d,m,n)  
#number=10
#a=Array(number,sint)    
#b=Array(number,sint)  
#c=Array(number,sint)
#mx=Matrix(2,number,sint) 
#a.input_from(1)
#b.assign([1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0])
#mx.input_from(0)
#ss,tt=all_attr(b,mx,a,2,number)
#print_ln("A:%s and T:%s",ss.reveal(),tt.reveal())
#AID,Threshold,NID,b=inner_node(3,mx,a,b,a,number,2)
#print_ln("A:%s and T:%s xk:%s b:%s",AID.reveal(),Threshold.reveal(),NID.reveal(),b.reveal()) 
#nid,p,q=leaf_node(3,b,a,mx[0],number)
#print_ln("NID:%s p:%s q:%s",nid.reveal(),p.reveal(),q.reveal())
#AIDM1,ThresholdM1,NIDM1,LNIDM1,LableP1,LableQ1=regression_tree(mx,a,2,2,number)
#r1,r2=decision_predict(AIDM1,ThresholdM1,LableP1,LableQ1,mx,2,2,number)
#print_ln("r1:%s r2:%s",r1.reveal(),r2.reveal())
#
