#####################################################################
# 
# 参考：https://github.com/Taiki-azrs/RaspiGPGPU_guide/tree/master/chap05
#####################################################################
#coding:utf-8
from time import clock_gettime, CLOCK_MONOTONIC
import numpy as np
from PIL import Image, ImageFilter

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
    #                    thy=2
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
    #==================================
    iidx=rf50
    jidx=rf51
    kidx=rf52
    istp=rf53
    jstp=rf54
    a_cur=rf55
    b_cur=rf56
    c_cur=rf57
    simd_stp=rf58
    mov(simd_stp,1)
    shl(simd_stp,simd_stp,6)
    #istp=16*
    mov(iidx,0)
    with loop as iloop:
        #iidx x 16 x A_STR + (A_ADDR + (eidx x A_STR))
        eidx(a_cur)
        rotate(broadcast,r2,-A_STR)
        umul24(a_cur,a_cur,r5)
        mov(r0,1)
        shl(r0,r0,4)
        umul24(r0,r0,r5)
        umul24(r0,r0,iidx)
        rotate(broadcast,r2,-A_ADDR)
        add(a_cur,a_cur,r5)
        add(a_cur,a_cur,r0)
        mov(jidx,0)
        with loop as jloop:
            mov(kidx,0)
            #(eidx x 4  + B_ADDR) + jidx x 32 x 4
            eidx(b_cur)
            shl(b_cur,b_cur,2)
            mov(r0,1)
            shl(r0,r0,7)
            umul24(r0,r0,jidx)
            rotate(broadcast,r2,-B_ADDR)
            add(b_cur,b_cur,r5)
            add(b_cur,b_cur,r0)
            
            with loop as kloop:
                add(kidx,kidx,1)
                mov(tmua,a_cur,sig=thrsw)
                nop()
                nop()
                nop(sig=ldtmu(r3))
                for lj in range(2):
                    stp=lj*16
                    mov(tmua,b_cur,sig=thrsw)
                    nop()
                    nop()
                    nop(sig=ldtmu(r4))
                    for li in range(16):
                        rotate(broadcast,r3,-li)
                        fmul(r0,r5,r4)
                        fadd(rf[stp+li],rf[stp+li],r0)
                        #add 1row and colmun
                    add(b_cur,b_cur,simd_stp)

                add(a_cur,a_cur,4)
                #調整
                sub(b_cur,b_cur,simd_stp)
                #str-16
                rotate(broadcast,r2,-B_STR)
                sub(r5,r5,simd_stp)
                
                add(b_cur,b_cur,r5)
                rotate(broadcast,r2,-K_SIZE)
                sub(null,r5,kidx,cond = 'pushz')
                #mov(null,0,cond = 'pushz')
                kloop.b(cond='anyna')
                nop()
                nop()
                nop()

            #Write C
            #set C addr
            #iidx x 16 x B_STR 
            mov(r3,1)
            shl(r3,r3,4)
            rotate(broadcast,r2,-B_STR)
            umul24(c_cur,r3,r5)
            umul24(c_cur,c_cur,iidx)
            #jidx x 32 x 4
            shl(r3,r3,3)
            umul24(r3,r3,jidx)
            add(c_cur,r3,c_cur)
            rotate(broadcast,r2,-C_ADDR)
            add(c_cur,c_cur,r5)
            eidx(r3)
            shl(r3,r3,2)
            add(c_cur,c_cur,r3)
            rotate(broadcast,r2,-B_STR)
            for li in range(16):
                mov(tmud,rf[li])
                #mov(tmud,1.0)
                mov(tmua,c_cur)
                mov(rf[li],0.0)
                tmuwt()

                mov(tmud,rf[16+li])
                add(tmua,c_cur,simd_stp)
                mov(rf[16+li],0.0)
                tmuwt()
                add(c_cur,c_cur,r5)

            add(jidx,jidx,1)    
            rotate(broadcast,r2,-J_SIZE)
            sub(null,r5,jidx,cond = 'pushz')
            #mov(null,0,cond = 'pushz')
            jloop.b(cond='anyna')
            nop()
            nop()
            nop()
        #I_SIZE-iidx==0
        add(iidx,iidx,1)
        rotate(broadcast,r2,-I_SIZE)
        sub(null,r5,iidx,cond = 'pushz')
        #mov(null,0,cond = 'pushz')
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
    p=160
    q=768
    r=2304
    hblock=p/num_thy
    wblock=r/num_thx
    #16x16/1loop/1th
    j_idx=wblock/32
    i_idx=hblock/16
    k_idx=q

    
    with Driver() as drv:
        # params setting
        A = drv.alloc((1, p, q), dtype='float32')
        B = drv.alloc((q, r), dtype='float32')
        C = drv.alloc((1, p, r), dtype='float32')
        
        A[:] = np.random.rand(p,q)*0.1
        B[:] = np.random.rand(q,r)*0.1
        C[:] = 0
        
        # uniform setting
        unif = drv.alloc(16, dtype='uint32')
        unif[0] = A.address
        unif[1] = A.strides[-2]  
        unif[2] = B.address
        unif[3] = B.strides[-2]
        unif[4] = C.address
        unif[5] = hblock
        unif[6] = wblock*4 #float
        unif[7] = i_idx
        unif[8] = j_idx
        unif[9] = k_idx
        print(unif)
        code = drv.program(kernel, num_qpus=num_qpus)
        # Run the program
        start = getsec()
        drv.execute(code, unif.addresses()[0], thread=num_qpus)
        time_gpu = (getsec() - start)*1000.0
        C_ref=np.dot(A,B)
        def gflops(time):
            return p*q*r*2/(time/1000)/1024/1024/1024
        
        print(f'QPU:   {time_gpu:.4} msec')
        print(f'{gflops(time_gpu)}GFLOPS')
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
