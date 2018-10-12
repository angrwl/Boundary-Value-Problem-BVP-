def crout(aim1,aii,aip1,f): # crout factorization for tridiagonal matrices with diagonals aim1, aii, aip1
    n=len(aim1)
    lim1=zeros(n)
    lii=zeros(n)
    uip1=zeros(n)
    z=zeros(n)
    x=zeros(n)
# compute factorisation
    lii[0]=aii[0]
    uip1[0]=aip1[0]/lii[0]
    for i in range(1,n-1):
        lim1[i]=aim1[i]
        lii[i]=aii[i]-lim1[i]*uip1[i-1]
        uip1[i]=aip1[i]/lii[i]
    lim1[-1]=aim1[-1]
    lii[-1]=aii[-1]-lim1[-1]*uip1[-2]
# solve for x
    z[0]=f[0]/lii[0]
    for i in range(1,n):
       z[i]= (f[i]-lim1[i]*z[i-1])/lii[i]
    x[-1]=z[-1]
    for i in range(n-2,-1,-1):
       x[i]=z[i]-uip1[i]*x[i+1]
    return x
