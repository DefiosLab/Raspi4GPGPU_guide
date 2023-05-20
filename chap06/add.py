#coding:utf-8
from time import clock_gettime, CLOCK_MONOTONIC
import numpy as np
from PIL import Image, ImageFilter
import time
from videocore6.assembler import qpu
from videocore6.driver import Driver


def getsec():
    return clock_gettime(CLOCK_MONOTONIC)

def get_thid():
    tidx(r0)
    shr(r0, r0, 2)
    band(r0, r0, 0b1111)    

def read_add_write():
    mov(tmua,r1,sig=thrsw)
    mov(r0,1) # nop()
    shl(r0,r0,6) # nop()
    nop(sig=ldtmu(rf5))    
    mov(tmua,r3,sig=thrsw)
    add(r1,r1,r0)    # nop()
    add(r3,r3,r0)    # nop()
    nop(sig=ldtmu(rf6))
    fadd(rf0,rf5,rf6)
    mov(tmud,rf0)
    mov(tmua,rf1)
    add(rf1,rf1,r0)
    tmuwt()            
    
@qpu
def kernel(asm, num_qpus):
    A_ADDR=0
    B_ADDR=1
    C_ADDR=2
    PSIZE=3
    LOOP_NUM=4
    EDGE_MOD=5
    LOOP_NUM_LTH=6
    EDGE_MOD_LTH=7
    eidx(r0).mov(r2, 0)
    for idx in [A_ADDR, B_ADDR, C_ADDR, PSIZE, LOOP_NUM, EDGE_MOD, LOOP_NUM_LTH, EDGE_MOD_LTH]:
        nop(sig=ldunifrf(r5))
        sub(null, r0, idx, cond='pushz')
        mov(r2, r5, cond='ifa')
        
    if num_qpus == 1:
        mov(r0, 0)
    elif num_qpus == 8:
        get_thid() #r0にthread idを格納
    else:
        raise Exception('num_qpus must be 1 or 8')
    
    for i in range(64):
        mov(rf[i],0.0)

    sub(null,r0,7,cond='pushz')
    b(R.not_th_id7,cond='anyna')
    nop()
    nop()
    nop()
    eidx(r3)

    rotate(broadcast,r2,-LOOP_NUM_LTH)
    sub(null,r3,LOOP_NUM,cond='pushz')
    mov(r2,r5,cond='ifa')

    rotate(broadcast,r2,-EDGE_MOD_LTH)
    sub(null,r3,EDGE_MOD,cond='pushz')
    mov(r2,r5,cond='ifa')
           
    L.not_th_id7
    #set per thread data
    rotate(broadcast,r2,-PSIZE)
    umul24(r1,r5,r0)
    shl(r1,r1,2)
    nop()
    nop()
    eidx(r4)
    sub(null, r4,A_ADDR,cond='pushz')
    add(r2, r2, r1, cond='ifa')
    sub(null, r4,B_ADDR,cond='pushz')
    add(r2, r2, r1, cond='ifa')    
    sub(null, r4,C_ADDR,cond='pushz')
    add(r2, r2, r1, cond='ifa')    

    eidx(r0)
    shl(r0,r0,2)
    rotate(broadcast,r2,-A_ADDR)
    add(r1,r0,r5)
    rotate(broadcast,r2,-B_ADDR)
    add(r3,r0,r5)
    rotate(broadcast,r2,-C_ADDR)
    add(rf1,r0,r5)

    rotate(broadcast,r2,-LOOP_NUM)
    mov(null,r5,cond='pushz')
    b(R.jmp,cond='anya')
    nop()
    nop()
    nop()
    
    with loop as iloop:
        read_add_write()
        rotate(broadcast,r2,-LOOP_NUM)
        sub(r0,r5,1,cond = 'pushz')
        iloop.b(cond='anyna')
        eidx(r4)
        sub(null, r4,LOOP_NUM,cond='pushz')
        mov(r2, r0, cond='ifa')    
    L.jmp

    #SIMDの端数処理
    rotate(broadcast,r2,-EDGE_MOD)
    shl(r5,r5,2)
    sub(r1,r1,r5)
    sub(r3,r3,r5)
    sub(rf1,rf1,r5)
    read_add_write()
    



        
    L.end
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
    N=1024
    M=1024
    if N*M<=128:
        num_qpus = 1
    else:
        num_qpus = 8
    # C画像サイズ

    simd_width=16

    
    qpu_mod = (N*M)%num_qpus
    proc_size=int((N*M)/num_qpus)
    proc_size_lth = qpu_mod+proc_size
    loop_num_lth = int(proc_size_lth/simd_width)
    loop_num=int(proc_size/simd_width)
    edge_mod_lth = simd_width-proc_size_lth%simd_width
    edge_mod = simd_width-proc_size%simd_width
    with Driver() as drv:
        # params setting
        A = drv.alloc((N, M), dtype='float32')
        B = drv.alloc((N, M), dtype='float32')
        C = drv.alloc((N, M), dtype='float32')
        A[:] = np.random.rand(N,M)*0.1
        B[:] = np.random.rand(N,M)*0.1
        C[:] = 0
        
        # uniform setting
        unif = drv.alloc(16, dtype='uint32')
        unif[0] = A.address
        unif[1] = B.address
        unif[2] = C.address
        unif[3] = proc_size
        unif[4] = loop_num
        unif[5] = edge_mod
        unif[6] = loop_num_lth
        unif[7] = edge_mod_lth
        code = drv.program(kernel, num_qpus=num_qpus)
        # Run the program
        gpu_time=0
        for i in range(10):
            start = time.time()
            drv.execute(code, unif.addresses()[0], thread=num_qpus)
            end = time.time()
            gpu_time+=(end-start)*1000.0

        cpu_time=0
        for i in range(10):
            start = time.time()
            C_ref=A+B
            end = time.time()
            cpu_time+=(end-start)*1000.0
        print("cpu time:{}msec".format(cpu_time/10))
        print("gpu time:{}msec".format(gpu_time/10))
        print(C,C_ref)
        
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
