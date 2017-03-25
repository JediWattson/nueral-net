import math
from random import randint

#get the index of the max
def maxind(o):
    m = 0
    ind = 0
    d = sum(o)
    s = ''
    for i, out in enumerate(o):
	out = out/d
	s += str(out) + '\t'
        if out > m:
            m = out
            ind = i
    return ind

#initialize the matrices
def initial(W,B,I, i, h):
    swap = []
    for x in range(i):
        swap.append(1.00/float(x+2))

    for x in range(h):
	rand = []
	rs = swap[:]
        for y in range(i):
	    if len(rs) != 0:
	        s = randint(0,len(rs)-1)           
            	rand.append(rs[s])
		del rs[s]
        W.append(rand)

    rs = swap[:]
    for z in range(i):
        if len(rs) != 0:
	    s = randint(0,len(rs)-1)
    	    I.append(rs[s])
            del rs[s]

    swap = []
    for x in range(h):
        swap.append(1.00/float(x+2))        
    rs = swap[:]
    for z in range(h):
        if len(rs) != 0:
	    s = randint(0,len(rs)-1)
	    B.append(rs[s])
            del rs[s]


    return W, B, I

#matrix multiplication
def mmul(A,B):
    outMul = []
    for row in A:
        s = 0
        for i, a in enumerate(row):
            s += a*B[i]
        outMul.append(s)
    return outMul

#matrix addition
def madd(A,B):
    outAdd = []
    for i, a in enumerate(B):
       outAdd.append(a+A[i])
    return outAdd

#sigmoid function
def msig(A):
    outSig = []
    dirSig = []
    for a in A:
	s = 1/(1 + math.exp(-a))
        outSig.append(s)
	dirSig.append(s*(s-1))
    return outSig, dirSig

#backprop with weights
def bpropw(W,B,I,y,o,p):
    ld = []
    alpha = .1

    for hid, row in enumerate(zip(*W)):
        if hid == 0:
            for inp, wt in enumerate(row):
                if y == inp:
                    c = (o[inp] - 1)
                else:
                    c = o[inp]
                W[inp][hid] -= wt*c*p[inp]*alpha
		B[inp] -= B[inp]*c*alpha
                ld.append(c)
        else:
            for inp, wt in enumerate(row):
                W[inp][hid] -= ld[inp]*wt*alpha
                ld[inp] = c
    return W, B

def driver():
    inp = 800
    hid = 20
    z = 0
    y = [randint(0,inp-1)]
    i = y[0] -1 

    o1 = []
    o2 = []
    d1 = []
    d2 = []
   
    I = []

    W = []
    Bw = []
    W, Bw, I = initial(W, Bw, I, inp, hid)

    U = []
    Bu = []
    U, Bu, o2 = initial(U, Bu,[], hid, inp)
    while i != y[0]:
        z += 1 
	o1 = mmul(W, I)
	o1 = madd(o1, Bw)
	o1, d1 = msig(o1)

	o2 = mmul(U, o1)

	o2 = madd(o2, Bu)
	o2, d2 = msig(o2)

	i = maxind(o2)
	W,Bw = bpropw(W,Bw,I,y[0],o1, d1)
	U,Bu = bpropw(U,Bu,o1,y[0],o2, d2)
    print "iterations: ", z

if __name__ == "__main__":
    for x in range(220):
	print "step: ", x+1
        driver()



