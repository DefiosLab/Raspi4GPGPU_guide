#coding:utf-8
from time import clock_gettime, CLOCK_MONOTONIC
import numpy as np
from PIL import Image, ImageFilter
import time
from videocore6.assembler import qpu
from videocore6.driver import Driver
#np.set_printoptions(threshold=1000000)

def getsec():
    return clock_gettime(CLOCK_MONOTONIC)

@qpu
def kernel(asm, num_qpus):
    A_ADDR=0
    A_STR=1
    B_ADDR=2
    B_STR=3
    C_ADDR=4
    HBLOCK=5
    WBLOCK=6
    I_SIZE=7
    J_SIZE=8
    K_SIZE=9
    I_IDX=10
    J_IDX=11
    k_IDX=12

    eidx(r0).mov(r2, 0)
    for idx in [A_ADDR, A_STR, B_ADDR, B_STR, C_ADDR, HBLOCK, WBLOCK, I_SIZE, J_SIZE, K_SIZE]:
        nop(sig=ldunifrf(r5))
        sub(null, r0, idx, cond='pushz')
        mov(r2, r5, cond='ifa')
        
    if num_qpus == 1:
        mov(r0, 0)
    elif num_qpus == 8:
        tidx(r0)
        shr(r0, r0, 2)
        band(r0, r0, 0b1111)
    else:
        raise Exception('num_qpus must be 1 or 8')
    
    for i in range(64):
        mov(rf[i],0.0)
    #=======numqpu=8 to  thx=4========
    #                   thy=2
    #if(thx-4>=0){thx-=4}
    #B set
    
    eidx(r4)
    sub(r1,r0,4,cond='pushn')
    mov(r1,r0,cond='ifa')
    rotate(broadcast,r2,-WBLOCK)
    umul24(r1,r5,r1)
    sub(null, r4, B_ADDR, cond='pushz')
    add(r2, r2, r1, cond='ifa')    

    
    #numqpu%4
    #A set
    
    shr(r3,r0,2)
    rotate(broadcast,r2,-HBLOCK)
    umul24(r3,r5,r3)
    rotate(broadcast,r2,-B_STR) #for C
    umul24(r0,r3,r5)
    
    rotate(broadcast,r2,-A_STR) #for A
    umul24(r3,r5,r3)
    sub(null, r4, A_ADDR, cond='pushz')
    add(r2, r2, r3, cond='ifa')    

    #C set
    add(r1,r1,r0)
    sub(null, r4, C_ADDR, cond='pushz')
    add(r2, r2, r1, cond='ifa')        
    iidx=rf50
    jidx=rf51
    kidx=rf52
    istp=rf53
    jstp=rf54
    a_cur=rf55
    b_cur=rf56
    c_cur=rf57
    simd_stp=rf58
    ldi128=rf59
    ldi16=rf60
    mov(ldi128,1)
    shl(ldi128,ldi128,7)
    mov(ldi16,1)
    shl(ldi16,ldi16,4)
    #simd_stp = 16 x 4
    mov(simd_stp,1)
    shl(simd_stp,simd_stp,6)
    mov(iidx,0)

    with loop as iloop:
        #set a_cur
        #16 x iidx x A_STR x eidx + A_ADDR
        # mov(r0,1)
        # shl(r0,r0,4)
        umul24(r0,ldi16,iidx)
        eidx(a_cur)
        rotate(broadcast,r2,-A_STR)
        umul24(a_cur,a_cur,r5)
        umul24(r0,r5,r0)
        add(a_cur,a_cur,r0)
        rotate(broadcast,r2,-A_ADDR)
        add(a_cur,a_cur,r5)
        mov(jidx,0)
        with loop as jloop:
            #set b_cur
            #1 : 32 x 4(float) x jidx
            umul24(r0,ldi128,jidx)


            #2 : eidx x 4 + B_ADDR
            mov(kidx,0)
            eidx(b_cur)
            shl(b_cur,b_cur,2)
            rotate(broadcast,r2,-B_ADDR)
            add(b_cur,b_cur,r5)

            #1 + 2
            add(b_cur,b_cur,r0)

            with loop as kloop:
                mov(tmua,a_cur,sig=thrsw)
                add(a_cur,a_cur,4) #nop()
                add(kidx,kidx,1) #nop()
                nop(sig=ldtmu(r3))
                for lj in range(2):
                    stp = lj*16
                    mov(tmua,b_cur,sig=thrsw)
                    if lj==0:
                        add(b_cur,b_cur,simd_stp) #nop()
                    else:
                        nop()
                    nop()
                    nop(sig=ldtmu(r4))
                    rotate(broadcast,r4,0)
                    fmul(r0,r5,r3)                    
                    for li in range(15):
                        rotate(broadcast,r4,-(li+1))
                        fadd(rf[stp+li],rf[stp+li],r0).fmul(r0,r5,r3)
                    fadd(rf[stp+15],rf[stp+15],r0)
                

                rotate(broadcast,r2,-K_SIZE)
                sub(null,r5,kidx,cond='pushz')
                kloop.b(cond='anyna')
                sub(b_cur,b_cur,simd_stp) #nop()
                rotate(broadcast,r2,-B_STR) #nop()
                add(b_cur,b_cur,r5) #nop() 

            # 32 x 4(float) x jidx
            # mov(r0,1)
            # shl(r0,r0,4)
            umul24(r0,ldi16,iidx)
            eidx(c_cur)
            rotate(broadcast,r2,-B_STR)
            umul24(c_cur,c_cur,r5)
            umul24(r0,r5,r0)
            add(c_cur,c_cur,r0)
            
            # mov(r0,1)
            # shl(r0,r0,7)
            umul24(r0,ldi128,jidx)            
            rotate(broadcast,r2,-C_ADDR)
            add(c_cur,c_cur,r5)
            add(c_cur,c_cur,r0)
            for li in range(32):
                # fmul(r3,r3,2.0)
                # mov(tmud,r3)
                mov(tmud,rf[li])
                mov(tmua,c_cur)
                add(c_cur,c_cur,4)
                mov(rf[li],0.0)
                tmuwt()
            rotate(broadcast,r2,-J_SIZE)
            add(jidx,jidx,1)
            sub(null,r5,jidx,cond = 'pushz')
            jloop.b(cond='anyna')
            rotate(broadcast,r2,-A_STR) #nop()
            sub(a_cur,a_cur,r5) #nop()
            nop()
        add(iidx,iidx,1)
        rotate(broadcast,r2,-I_SIZE)
        sub(null,r5,iidx,cond = 'pushz')
        iloop.b(cond='anyna')
        nop()
        nop()
        nop()
        
    barrierid(syncb, sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()
def main():
    num_thx=4
    num_thy=2    
    num_qpus = num_thx*num_thy
    # C画像サイズ
    p=1024
    q=1024
    r=1024
    hblock=p/num_thy
    wblock=r/num_thx
    #16x16/1loop/1th
    j_idx=wblock/32
    i_idx=hblock/16
    k_idx=q

    
    with Driver(data_area_size=1024*1024*1024+1024*1024*512) as drv:
        
        # params setting
        A = drv.alloc((p, q), dtype='float32')
        B = drv.alloc((q, r), dtype='float32')
        C = drv.alloc((p, r), dtype='float32')
        
        A[:] = np.random.rand(p,q)*0.1
        B[:] = np.random.rand(q,r)*0.1
        C[:] = 0

        # A[:]=np.arange(A.size).reshape(A.shape)
        # B[:]=np.arange(B.size).reshape(B.shape)        
        # uniform setting
        unif = drv.alloc(16, dtype='uint32')
        unif[0] = A.addresses()[0,0]
        unif[1] = A.strides[0]  
        unif[2] = B.addresses()[0,0]
        unif[3] = B.strides[0]
        unif[4] = C.addresses()[0,0]
        unif[5] = hblock
        unif[6] = wblock*4 #float
        unif[7] = i_idx
        unif[8] = j_idx
        unif[9] = k_idx
        code = drv.program(kernel, num_qpus=num_qpus)
        iteration = 10
        # Run the program
        gpu_time = 0
        for i in range(iteration):
            C[:]=0
            st = time.time()
            drv.execute(code, unif.addresses()[0], thread=num_qpus)
            ed =time.time()
            gpu_time += ed-st
        average_gpu_time = gpu_time / iteration

        
        cpu_time = 0
        for i in range(iteration):
            st = time.time()
            C_ref=np.dot(A,B)
            ed =time.time()
            cpu_time += ed-st
        average_cpu_time = cpu_time /iteration
            
        def gflops(time):
            return p * q * r * 2 / average_gpu_time * 1e-9
        
        print(f'GPU time:   {average_gpu_time*1000} msec')
        print(f'CPU time:   {average_cpu_time*1000} msec')
        print(f'{gflops(gpu_time)}GFLOPS')
        print('minimum absolute error: {:.4e}'.format(
            float(np.min(np.abs(C_ref - C)))))
        print('maximum absolute error: {:.4e}'.format(
            float(np.max(np.abs(C_ref - C)))))
        print('minimum relative error: {:.4e}'.format(
            float(np.min(np.abs((C_ref - C) / C_ref)))))
        print('maximum relative error: {:.4e}'.format(
            float(np.max(np.abs((C_ref - C) / C_ref)))))

if __name__ == '__main__':
    main()
