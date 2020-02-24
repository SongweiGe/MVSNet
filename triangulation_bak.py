import time
import torch
import numpy as np

try:
    from rpcm.rpc_model import rpc_from_geotiff
except:
    pass

def RPCnormalization(undata,offset,scale):
    return (undata-offset)/scale


def RPCunnormalization(normdata,offset,scale):
    return normdata*scale+offset


def RPCsinglepolynomialderivativeX_matrix(p,X,Y,Z):
    npoints = len(X)
    # monomial=[1 X Y Z X*Y X*Z Y*Z X**2 Y**2 Z**2 X*Y*Z X**3 X*Y**2 X*Z**2 X**2*Y Y**3 Y*Z**2 X**2*Z Y**2*Z Z**3]
    monomial = torch.stack([torch.zeros(npoints), torch.ones(npoints), torch.zeros(npoints), torch.zeros(npoints), Y, Z, torch.zeros(npoints), 
                2*X, torch.zeros(npoints), torch.zeros(npoints), Y*Z, 3*X**2, Y**2, Z**2, 2*X*Y, torch.zeros(npoints), 
                torch.zeros(npoints), 2*X*Z, torch.zeros(npoints), torch.zeros(npoints)]).double().cuda() # 20 x npoints
    sol = torch.sum(p.reshape(-1, 1)*monomial, 0)
    return sol


def RPCsinglepolynomialderivativeY_matrix(p,X,Y,Z):
    npoints = len(X)
    # monomial=[1 X Y Z X*Y X*Z Y*Z X**2 Y**2 Z**2 X*Y*Z X**3 X*Y**2 X*Z**2 X**2*Y Y**3 Y*Z**2 X**2*Z Y**2*Z Z**3]
    monomial = torch.stack([torch.zeros(npoints), torch.zeros(npoints), torch.ones(npoints), torch.zeros(npoints), X, torch.zeros(npoints), Z, 
            torch.zeros(npoints), 2*Y, torch.zeros(npoints), X*Z, torch.zeros(npoints), X*2*Y, torch.zeros(npoints), 
            X**2, 3*Y**2, Z**2, torch.zeros(npoints), 2*Y*Z, torch.zeros(npoints)]).double().cuda()
    sol = torch.sum(p.reshape(-1, 1)*monomial, 0)
    return sol
   

def RPCsinglepolynomialderivativeZ_matrix(p,X,Y,Z):
    npoints = len(X)
    # monomial=[1 X Y Z X*Y X*Z Y*Z X**2 Y**2 Z**2 X*Y*Z X**3 X*Y**2 X*Z**2 X**2*Y Y**3 Y*Z**2 X**2*Z Y**2*Z Z**3]
    monomial = torch.stack([torch.zeros(npoints), torch.zeros(npoints), torch.zeros(npoints), torch.ones(npoints), torch.zeros(npoints), X, Y, 
            torch.zeros(npoints), torch.zeros(npoints), 2*Z, X*Y, torch.zeros(npoints), torch.zeros(npoints), X*2*Z, 
            torch.zeros(npoints), torch.zeros(npoints), Y*2*Z, X**2, Y**2, 3*Z**2]).double().cuda()
    sol = torch.sum(p.reshape(-1, 1)*monomial, 0)
    return sol 


def RPCsinglepolynomial_matrix(p,X,Y,Z):
    npoints = len(X)
    # import ipdb;ipdb.set_trace()
    monomial = torch.stack([torch.ones(npoints), X, Y, Z, X*Y, X*Z, Y*Z, X**2, Y**2, Z**2, X*Y*Z, 
                X**3, X*Y**2, X*Z**2, X**2*Y, Y**3, Y*Z**2, X**2*Z, Y**2*Z, Z**3]).double().cuda()
    sol = torch.sum(p.reshape(-1, 1)*monomial, 0)
    return sol


def RPCforwardform_matrix(p,q,X,Y,Z):
    npoints = len(X)
    # num = torch.zeros((npoints,20))
    # den = torch.zeros((npoints,20))
    monomial = torch.stack([torch.ones(npoints), X, Y, Z, X*Y, X*Z, Y*Z, X**2, Y**2, Z**2, X*Y*Z, 
                X**3, X*Y**2, X*Z**2, X**2*Y, Y**3, Y*Z**2, X**2*Z, Y**2*Z, Z**3]).double().cuda()
    num = torch.sum(p.reshape(-1, 1)*monomial, 0)
    den = torch.sum(q.reshape(-1, 1)*monomial, 0)
    if (den == 0).all():
        den = 1
    pixel_coordinate=num/den
    return pixel_coordinate


def triangulationRPC_matrix(ru1, cu1, ru2, cu2, rpc1, rpc2, verbose, inverse_bs=100):
    begin = time.time()
    npoints = len(ru1)
    print('start triangulation with number of points: %d'%npoints)
    #  setup Parameters based on the notation
    p1_1 = torch.cuda.DoubleTensor(np.array(rpc1.row_num, dtype=np.float64))
    p2_1 = torch.cuda.DoubleTensor(np.array(rpc1.row_den, dtype=np.float64))
    p3_1 = torch.cuda.DoubleTensor(np.array(rpc1.col_num, dtype=np.float64))
    p4_1 = torch.cuda.DoubleTensor(np.array(rpc1.col_den, dtype=np.float64))
    p1_2 = torch.cuda.DoubleTensor(np.array(rpc2.row_num, dtype=np.float64))
    p2_2 = torch.cuda.DoubleTensor(np.array(rpc2.row_den, dtype=np.float64))
    p3_2 = torch.cuda.DoubleTensor(np.array(rpc2.col_num, dtype=np.float64))
    p4_2 = torch.cuda.DoubleTensor(np.array(rpc2.col_den, dtype=np.float64))

    # r1, c1, r2, c2

    r1 = (ru1-rpc1.row_offset)/rpc1.row_scale
    c1 = (cu1-rpc1.col_offset)/rpc1.col_scale
    r2 = (ru2-rpc2.row_offset)/rpc2.row_scale
    c2 = (cu2-rpc2.col_offset)/rpc2.col_scale
    # normalization parameters
    Ys_1 = rpc1.lat_scale
    Yo_1 = rpc1.lat_offset
    Xs_1 = rpc1.lon_scale
    Xo_1 = rpc1.lon_offset
    Zs_1 = rpc1.alt_scale
    Zo_1 = rpc1.alt_offset
    Ys_2 = rpc2.lat_scale
    Yo_2 = rpc2.lat_offset
    Xs_2 = rpc2.lon_scale
    Xo_2 = rpc2.lon_offset
    Zs_2 = rpc2.alt_scale
    Zo_2 = rpc2.alt_offset
    # 1st iteration only linear model:
    # camera 1
    # row
    a0_1=p1_1[0]*Zs_1*Ys_1*Xs_1 - p1_1[3]*Zo_1*Ys_1*Xs_1-p1_1[2]*Zs_1*Yo_1*Xs_1-p1_1[1]*Zs_1*Ys_1*Xo_1
    a1_1=p1_1[3]*Ys_1*Xs_1
    a2_1=p1_1[2]*Zs_1*Xs_1
    a3_1=p1_1[1]*Zs_1*Ys_1
    b0_1=p2_1[0]*Zs_1*Ys_1*Xs_1-p2_1[3]*Zo_1*Ys_1*Xs_1-p2_1[2]*Zs_1*Yo_1*Xs_1-p2_1[1]*Zs_1*Ys_1*Xo_1
    b1_1=p2_1[3]*Ys_1*Xs_1
    b2_1=p2_1[2]*Zs_1*Xs_1
    b3_1=p2_1[1]*Zs_1*Ys_1
    # col
    c0_1=p3_1[0]*Zs_1*Ys_1*Xs_1 - p3_1[3]*Zo_1*Ys_1*Xs_1-p3_1[2]*Zs_1*Yo_1*Xs_1-p3_1[1]*Zs_1*Ys_1*Xo_1
    c1_1=p3_1[3]*Ys_1*Xs_1
    c2_1=p3_1[2]*Zs_1*Xs_1
    c3_1=p3_1[1]*Zs_1*Ys_1
    d0_1=p4_1[0]*Zs_1*Ys_1*Xs_1-p4_1[3]*Zo_1*Ys_1*Xs_1-p4_1[2]*Zs_1*Yo_1*Xs_1-p4_1[1]*Zs_1*Ys_1*Xo_1
    d1_1=p4_1[3]*Ys_1*Xs_1
    d2_1=p4_1[2]*Zs_1*Xs_1
    d3_1=p4_1[1]*Zs_1*Ys_1
    # camera 2
    # row
    a0_2=p1_2[0]*Zs_2*Ys_2*Xs_2 - p1_2[3]*Zo_2*Ys_2*Xs_2-p1_2[2]*Zs_2*Yo_2*Xs_2-p1_2[1]*Zs_2*Ys_2*Xo_2
    a1_2=p1_2[3]*Ys_2*Xs_2
    a2_2=p1_2[2]*Zs_2*Xs_2
    a3_2=p1_2[1]*Zs_2*Ys_2
    b0_2=p2_2[0]*Zs_2*Ys_2*Xs_2-p2_2[3]*Zo_2*Ys_2*Xs_2-p2_2[2]*Zs_2*Yo_2*Xs_2-p2_2[1]*Zs_2*Ys_2*Xo_2
    b1_2=p2_2[3]*Ys_2*Xs_2
    b2_2=p2_2[2]*Zs_2*Xs_2
    b3_2=p2_2[1]*Zs_2*Ys_2
    # col
    c0_2=p3_2[0]*Zs_2*Ys_2*Xs_2 - p3_2[3]*Zo_2*Ys_2*Xs_2-p3_2[2]*Zs_2*Yo_2*Xs_2-p3_2[1]*Zs_2*Ys_2*Xo_2
    c1_2=p3_2[3]*Ys_2*Xs_2
    c2_2=p3_2[2]*Zs_2*Xs_2
    c3_2=p3_2[1]*Zs_2*Ys_2
    d0_2=p4_2[0]*Zs_2*Ys_2*Xs_2-p4_2[3]*Zo_2*Ys_2*Xs_2-p4_2[2]*Zs_2*Yo_2*Xs_2-p4_2[1]*Zs_2*Ys_2*Xo_2
    d1_2=p4_2[3]*Ys_2*Xs_2
    d2_2=p4_2[2]*Zs_2*Xs_2
    d3_2=p4_2[1]*Zs_2*Ys_2
    # ITERATION 1
    print("Preparation time = %.3fs"%(time.time() - begin))

    A=torch.stack([torch.stack([a1_1-r1*b1_1, a2_1-r1*b2_1, a3_1-r1*b3_1]),
        torch.stack([c1_1-c1*d1_1, c2_1-c1*d2_1, c3_1-c1*d3_1]),
        torch.stack([a1_2-r2*b1_2, a2_2-r2*b2_2, a3_2-r2*b3_2]),
        torch.stack([c1_2-c2*d1_2, c2_2-c2*d2_2, c3_2-c2*d3_2])]).permute(2, 0, 1) # 4 x 4 x npoints
    b=torch.stack([r1*b0_1-a0_1,c1*d0_1-c0_1,r2*b0_2-a0_2,c2*d0_2-c0_2]).transpose(1, 0).view(-1, 4, 1) # 4 x npoints
    DeltaXu = torch.zeros(npoints)
    DeltaYu = torch.zeros(npoints)
    DeltaZu = torch.zeros(npoints)

    print("Preparation initialization time = %.3fs"%(time.time() - begin))

    for i in range(npoints//inverse_bs):
        # print(i)
        id_min = i*inverse_bs
        id_max = np.min([(i+1)*inverse_bs, npoints])
        A_temp = A[id_min:id_max]
        b_temp = b[id_min:id_max]
        try:
            LSsol= torch.matmul(torch.matmul(torch.inverse(torch.matmul(A_temp.transpose(2, 1), A_temp)), A_temp.transpose(2, 1)),b_temp)
        except:
            import ipdb;ipdb.set_trace()
        DeltaXu[id_min:id_max]=LSsol[:, 2, 0]
        DeltaYu[id_min:id_max]=LSsol[:, 1, 0]
        DeltaZu[id_min:id_max]=LSsol[:, 0, 0]
        print("Inverse step %d time = %.3fs"%(i, time.time() - begin))

    print("Initialization time = %.3fs"%(time.time() - begin))
    # torch.linalg.lstsq(A, b)
    # torch.matmul(A, LSsol)-b

    # DeltaZu=10
    # next iterations
    Niter=1
    NormXold=0
    NormYold=0
    NormZold=0
    Xunew_1=DeltaXu+0
    Yunew_1=DeltaYu+0
    Zunew_1=DeltaZu+0
    Xunew_2=DeltaXu+0
    Yunew_2=DeltaYu+0
    Zunew_2=DeltaZu+0
    NormupdateX_1=RPCnormalization(DeltaXu,Xo_1,Xs_1)    
    NormupdateY_1=RPCnormalization(DeltaYu,Yo_1,Ys_1)
    NormupdateZ_1=RPCnormalization(DeltaZu,Zo_1,Zs_1)
    NormupdateX_2=RPCnormalization(DeltaXu,Xo_2,Xs_2)
    NormupdateY_2=RPCnormalization(DeltaYu,Yo_2,Ys_2)
    NormupdateZ_2=RPCnormalization(DeltaZu,Zo_2,Zs_2)
    NormXnew_1=NormXold+NormupdateX_1
    NormYnew_1=NormYold+NormupdateY_1
    NormZnew_1=NormZold+NormupdateZ_1
    NormXnew_2=NormXold+NormupdateX_2
    NormYnew_2=NormYold+NormupdateY_2
    NormZnew_2=NormZold+NormupdateZ_2
    error_residual_old=10e10
    cnt=0
    for it in range(Niter):
        # import ipdb;ipdb.set_trace()
        # system eqs
        # partial derivatives
        # 1st camera
        # input needs to be normalized but the algorithm runs with
        # unnormalized updates (derivatives chain rule)
        pol1 = RPCsinglepolynomial_matrix(p1_1,NormXnew_1,NormYnew_1,NormZnew_1)
        pol2 = RPCsinglepolynomial_matrix(p2_1,NormXnew_1,NormYnew_1,NormZnew_1)
        pol3 = RPCsinglepolynomial_matrix(p3_1,NormXnew_1,NormYnew_1,NormZnew_1)
        pol4 = RPCsinglepolynomial_matrix(p4_1,NormXnew_1,NormYnew_1,NormZnew_1)
        print("Preparation 0 step %d time = %.3fs"%(it, time.time() - begin))
        pol1x = RPCsinglepolynomialderivativeX_matrix(p1_1,NormXnew_1,NormYnew_1,NormZnew_1)
        print("Preparation 0 step %d time = %.3fs"%(it, time.time() - begin))
        pol1y = RPCsinglepolynomialderivativeY_matrix(p1_1,NormXnew_1,NormYnew_1,NormZnew_1)
        pol1z = RPCsinglepolynomialderivativeZ_matrix(p1_1,NormXnew_1,NormYnew_1,NormZnew_1)
        pol2x = RPCsinglepolynomialderivativeX_matrix(p2_1,NormXnew_1,NormYnew_1,NormZnew_1)
        pol2y = RPCsinglepolynomialderivativeY_matrix(p2_1,NormXnew_1,NormYnew_1,NormZnew_1)
        pol2z = RPCsinglepolynomialderivativeZ_matrix(p2_1,NormXnew_1,NormYnew_1,NormZnew_1)
        pol3x = RPCsinglepolynomialderivativeX_matrix(p3_1,NormXnew_1,NormYnew_1,NormZnew_1)
        pol3y = RPCsinglepolynomialderivativeY_matrix(p3_1,NormXnew_1,NormYnew_1,NormZnew_1)
        pol3z = RPCsinglepolynomialderivativeZ_matrix(p3_1,NormXnew_1,NormYnew_1,NormZnew_1)
        pol4x = RPCsinglepolynomialderivativeX_matrix(p4_1,NormXnew_1,NormYnew_1,NormZnew_1)
        pol4y = RPCsinglepolynomialderivativeY_matrix(p4_1,NormXnew_1,NormYnew_1,NormZnew_1)
        pol4z = RPCsinglepolynomialderivativeZ_matrix(p4_1,NormXnew_1,NormYnew_1,NormZnew_1)
        pdXrow_1 = (pol1x*pol2-pol1*pol2x)/(pol2**2)
        pdYrow_1 = (pol1y*pol2-pol1*pol2y)/(pol2**2)
        pdZrow_1 = (pol1z*pol2-pol1*pol2z)/(pol2**2)
        pdXcol_1 = (pol3x*pol4-pol3*pol4x)/(pol4**2)
        pdYcol_1 = (pol3y*pol4-pol3*pol4y)/(pol4**2)
        pdZcol_1 = (pol3z*pol4-pol3*pol4z)/(pol4**2)
        print("Preparation 1 step %d time = %.3fs"%(it, time.time() - begin))
        # 2nd camera
        pol1 = RPCsinglepolynomial_matrix(p1_2,NormXnew_2,NormYnew_2,NormZnew_2)
        pol2 = RPCsinglepolynomial_matrix(p2_2,NormXnew_2,NormYnew_2,NormZnew_2)
        pol3 = RPCsinglepolynomial_matrix(p3_2,NormXnew_2,NormYnew_2,NormZnew_2)
        pol4 = RPCsinglepolynomial_matrix(p4_2,NormXnew_2,NormYnew_2,NormZnew_2)
        pol1x = RPCsinglepolynomialderivativeX_matrix(p1_2,NormXnew_2,NormYnew_2,NormZnew_2)
        pol1y = RPCsinglepolynomialderivativeY_matrix(p1_2,NormXnew_2,NormYnew_2,NormZnew_2)
        pol1z = RPCsinglepolynomialderivativeZ_matrix(p1_2,NormXnew_2,NormYnew_2,NormZnew_2)
        pol2x = RPCsinglepolynomialderivativeX_matrix(p2_2,NormXnew_2,NormYnew_2,NormZnew_2)
        pol2y = RPCsinglepolynomialderivativeY_matrix(p2_2,NormXnew_2,NormYnew_2,NormZnew_2)
        pol2z = RPCsinglepolynomialderivativeZ_matrix(p2_2,NormXnew_2,NormYnew_2,NormZnew_2)
        pol3x = RPCsinglepolynomialderivativeX_matrix(p3_2,NormXnew_2,NormYnew_2,NormZnew_2)
        pol3y = RPCsinglepolynomialderivativeY_matrix(p3_2,NormXnew_2,NormYnew_2,NormZnew_2)
        pol3z = RPCsinglepolynomialderivativeZ_matrix(p3_2,NormXnew_2,NormYnew_2,NormZnew_2)
        pol4x = RPCsinglepolynomialderivativeX_matrix(p4_2,NormXnew_2,NormYnew_2,NormZnew_2)
        pol4y = RPCsinglepolynomialderivativeY_matrix(p4_2,NormXnew_2,NormYnew_2,NormZnew_2)
        pol4z = RPCsinglepolynomialderivativeZ_matrix(p4_2,NormXnew_2,NormYnew_2,NormZnew_2)
        pdXrow_2 = (pol1x*pol2-pol1*pol2x)/(pol2**2)
        pdYrow_2 = (pol1y*pol2-pol1*pol2y)/(pol2**2)
        pdZrow_2 = (pol1z*pol2-pol1*pol2z)/(pol2**2)
        pdXcol_2 = (pol3x*pol4-pol3*pol4x)/(pol4**2)
        pdYcol_2 = (pol3y*pol4-pol3*pol4y)/(pol4**2)
        pdZcol_2 = (pol3z*pol4-pol3*pol4z)/(pol4**2)
        print("Preparation 2 step %d time = %.3fs"%(it, time.time() - begin))
        # build Jacobian A:
        A = torch.stack([torch.stack([pdXrow_1/Xs_1, pdYrow_1/Ys_1, pdZrow_1/Zs_1]),
            torch.stack([pdXcol_1/Xs_1, pdYcol_1/Ys_1, pdZcol_1/Zs_1]),
            torch.stack([pdXrow_2/Xs_2, pdYrow_2/Ys_2, pdZrow_2/Zs_2]),
            torch.stack([pdXcol_2/Xs_2, pdYcol_2/Ys_2, pdZcol_2/Zs_2])]).permute(2, 0, 1)
        r1_hat = RPCforwardform_matrix(p1_1,p2_1,NormXnew_1,NormYnew_1,NormZnew_1)
        c1_hat = RPCforwardform_matrix(p3_1,p4_1,NormXnew_1,NormYnew_1,NormZnew_1)
        r2_hat = RPCforwardform_matrix(p1_2,p2_2,NormXnew_2,NormYnew_2,NormZnew_2)
        c2_hat = RPCforwardform_matrix(p3_2,p4_2,NormXnew_2,NormYnew_2,NormZnew_2)
        b = torch.stack([r1-r1_hat, c1-c1_hat, r2-r2_hat, c2-c2_hat]).transpose(1, 0).view(-1, 4, 1)
        # bb=b.*[scale_offsets_1(9)scale_offsets_1(10)scale_offsets_2(9)scale_offsets_2(10)]
        # solution 
        print("Preparation step %d time = %.3fs"%(it, time.time() - begin))
        for i in range(npoints//inverse_bs):
            # print(i)
            id_min = i*inverse_bs
            id_max = np.min([(i+1)*inverse_bs, npoints])
            A_temp = A[id_min:id_max]
            b_temp = b[id_min:id_max]
            try:
                LSsol= torch.matmul(torch.matmul(torch.inverse(torch.matmul(A_temp.transpose(2, 1), A_temp)), 
                    A_temp.transpose(2, 1)),b_temp)
            except:
                import ipdb;ipdb.set_trace()
            DeltaXu[id_min:id_max]=LSsol[:, 0, 0]
            DeltaYu[id_min:id_max]=LSsol[:, 1, 0]
            DeltaZu[id_min:id_max]=LSsol[:, 2, 0]
        print("Step %d time = %.3fs"%(it, time.time() - begin))
        import ipdb;ipdb.set_trace()
        # for i in range(npoints):
        #     # import ipdb;ipdb.set_trace()
        #     LSsol= torch.matmul(torch.matmul(torch.inverse(torch.matmul(A[:, :, i].transpose(1,0), A[:, :, i])), A[:, :, i].transpose(1,0)),b[:, i])
        #     DeltaXu[i]=LSsol[0]
        #     DeltaYu[i]=LSsol[1]
        #     DeltaZu[i]=LSsol[2]
        # sanity check torch.matmul(A, LSsol)-b
        # if it == 1:
        #     import ipdb;ipdb.set_trace()
        # update unnormalized
        Xuold = Xunew_1
        Yuold = Yunew_1
        Zuold = Zunew_1 
        # import ipdb;ipdb.set_trace()
        Xunew_1 = Xunew_1 + DeltaXu
        Yunew_1 = Yunew_1 + DeltaYu
        Zunew_1 = Zunew_1 + DeltaZu
        Xunew_2 = Xunew_2 + DeltaXu
        Yunew_2 = Yunew_2 + DeltaYu
        Zunew_2 = Zunew_2 + DeltaZu
        error_residual = torch.mean(torch.sqrt(DeltaXu**2+DeltaYu**2+DeltaZu**2))        
        # normalize updated coordinates
        NormXnew_1=RPCnormalization(Xunew_1,Xo_1,Xs_1)
        NormYnew_1=RPCnormalization(Yunew_1,Yo_1,Ys_1)
        NormZnew_1=RPCnormalization(Zunew_1,Zo_1,Zs_1)
        NormXnew_2=RPCnormalization(Xunew_2,Xo_2,Xs_2)
        NormYnew_2=RPCnormalization(Yunew_2,Yo_2,Ys_2)
        NormZnew_2=RPCnormalization(Zunew_2,Zo_2,Zs_2)
        # convert degrees to UTM (m)
        # [Yunew_1_m,Xunew_1_m,utmzone] = deg2utm(Yunew_1, Xunew_1)
        # [Yuold_m,Xuold_m,utmzone] = deg2utm(Yuold, Xuold)
        # error_residual_m = sqrt((Xuold_m-Xunew_1_m)^2+(Yuold_m-Yunew_1_m)^2+DeltaZu^2)
        # error calculation
        r1_est =  RPCforwardform_matrix(p1_1,p2_1,NormXnew_1,NormYnew_1,NormZnew_1)
        c1_est =  RPCforwardform_matrix(p3_1,p4_1,NormXnew_1,NormYnew_1,NormZnew_1)
        r2_est =  RPCforwardform_matrix(p1_2,p2_2,NormXnew_2,NormYnew_2,NormZnew_2)
        c2_est =  RPCforwardform_matrix(p3_2,p4_2,NormXnew_2,NormYnew_2,NormZnew_2)
        # unnormalize        
        r1_est = RPCunnormalization(r1_est,rpc1.row_scale,rpc1.row_offset)
        c1_est = RPCunnormalization(c1_est,rpc1.col_scale,rpc1.col_offset)
        r2_est = RPCunnormalization(r2_est,rpc2.row_scale,rpc2.row_offset)
        c2_est = RPCunnormalization(c2_est,rpc2.col_scale,rpc2.col_offset)
        # r1_est = RPCunnormalization(r1_est,scale_offsets_1[7-1],scale_offsets_1[9-1])
        # c1_est = RPCunnormalization(c1_est,scale_offsets_1[8-1],scale_offsets_1[10-1])
        # r2_est = RPCunnormalization(r2_est,scale_offsets_2[7-1],scale_offsets_2[9-1])
        # c2_est = RPCunnormalization(c2_est,scale_offsets_2[8-1],scale_offsets_2[10-1])
        error_pixel_residual_1 = torch.mean(torch.sqrt((ru1-r1_est)**2+(cu1-c1_est)**2))
        error_pixel_residual_2 = torch.mean(torch.sqrt((ru2-r2_est)**2+(cu2-c2_est)**2))
        print("verbose time = %.3fs"%(time.time() - begin))
        if verbose:
            print('previous 3D point:%.10f,%.10f,%.10f updatedCAM1:%.10f,%.10f,%.10f updatedCAM2:%.10f,%.10f,%.10f \n'%
                (Xuold,Yuold,Zuold,Xunew_1,Yunew_1,Zunew_1,Xunew_2,Yunew_2,Zunew_2))
            print('original pixel pointCAM1:%.10f,%.10f updatedCAM1:%.10f,%.10f \n'%
                (ru1,cu1,r1_est,c1_est))
            # print('original pixel pointCAM2:%.10f,%.10f updatedCAM2:%.10f,%.10f \n'%
            #     (ru2,cu2,r2_est,c2_est))
            # print('Unnorm error3D:%.15f DeltaYu_m:%.10f DeltaXu_m:%.10f DeltaZu:%.10f at iter %d\n'%
            #     (error_residual_m,abs(Xuold_m-Xunew_1_m),abs(Yuold_m-Yunew_1_m),DeltaZu,it))
            print('Unnorm update DeltaZu:%.10f at iter %d\n'%(DeltaZu,it))
            print('Unnorm error_residual_m:%.10f error_pixel_residual_1:%.4f error_pixel_residual_2:%.4f at iter %d\n'%
                (error_residual,error_pixel_residual_1,error_pixel_residual_2,it))
        Xu = Xunew_1*0.5+Xunew_2*0.5
        Yu = Yunew_1*0.5+Yunew_2*0.5
        Zu = Zunew_1*0.5+Zunew_2*0.5        
        error_residual_old=error_residual
    # import ipdb;ipdb.set_trace()
    return Xu,Yu,Zu,error_residual,error_pixel_residual_1*0.5+error_pixel_residual_2*0.5


if __name__ == '__main__':
    geotiff_file = os.path.join(data_path, filename)
    geotiff_file = '02APR15WV031000015APR02134718-P1BS-500497282050_01_P001_________AAE_0AAAAABPABJ0.NTF.tif'
    rpc = rpc_from_geotiff(geotiff_file)