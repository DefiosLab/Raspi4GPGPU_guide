import onnx
import numpy as np
from videocore6.assembler import qpu
from videocore6.driver import Driver
import math
from .layer_base import Layer
class MatMul_gpu(Layer):
    def __init__(self,model,node,tensor_data,use_gpu=False,drv=None):
        super().__init__(model,node,tensor_data,use_gpu)
        self.drv=drv
        self.simd_width=16        
        self.input_data1 = self.alloc_gpu_for_initializer(self.input_name[0])
        self.input_data2 = self.alloc_gpu_for_initializer(self.input_name[1])
        self.output_data = self.tensor_data[self.output_name]
        data1_shape=list(self.input_data1.shape)
        data2_shape=list(self.input_data2.shape)
        output_shape=list(self.output_data.shape)
        self.p,self.q = data1_shape[-2:]
        self.q,self.r = data2_shape[-2:]

        if self.p < 32 or self.r < 128:
            num_thx = 1
            num_thy = 1
        else:
            num_thx=4
            num_thy=2
            
        assert self.p >= 16 and self.r >= 32, "The matrix size must satisfy p <= 16 and r <= 32."
        
        self.num_qpus = num_thx*num_thy


        #1スレッドが処理する範囲
        hblock=int(math.ceil(self.p/num_thy))
        wblock=int(math.ceil(self.r/num_thx))

        #スレッド分割の端数
        #GPUカーネル内で以下の処理を行う
        # if thread_id.x == 3:
        #     WBLOCK = WBLOCK - frac_w
        # if thread_id.y == 1:
        #     HBLOCK = HBLOCK - frac_h
        frac_w = (wblock * num_thx - self.r) * 4
        frac_h = (hblock * num_thy - self.p)

        #ループ回数
        j_idx=int(math.ceil(self.r / (32.0 * num_thx)))
        i_idx=int(math.ceil(self.p / (16.0 * num_thy)))
        k_idx=self.q
        
        #3次元対応
        if len(data1_shape) <3:
            data1_shape.insert(0,1)
        if len(data2_shape) <3:
            data2_shape.insert(0,1)            
        inp1_range = range(data1_shape[-3])
        inp2_range = range(data2_shape[-3])
        c1 = data1_shape[-3]
        c2 = data2_shape[-3]
        c_idx = max(c1,c2)
        inp1_size = self.p * self.q * 4 if c1 > 1 else 0
        inp2_size = self.q * self.r * 4 if c2 > 1 else 0
        out_size = self.p * self.r * 4 

        wblock = int(wblock * 4) #float = 4bytes
        self.output_data[:]=0
        self.unif = self.drv.alloc(16, dtype='uint32')
        self.unif[0] = self.input_data1.address
        self.unif[1] = self.input_data1.strides[-2]  
        self.unif[2] = self.input_data2.address
        self.unif[3] = self.input_data2.strides[-2]
        self.unif[4] = self.output_data.address
        self.unif[5] = hblock
        self.unif[6] = wblock
        self.unif[7] = i_idx
        self.unif[8] = j_idx
        self.unif[9] = k_idx
        self.unif[10] = frac_w
        self.unif[11] = frac_h
        self.unif[12] = inp1_size
        self.unif[13] = inp2_size
        self.unif[14] = out_size
        self.unif[15] = c_idx
        self.code = self.drv.program(kernel, num_qpus=self.num_qpus)        
            
    def run(self):
        self.drv.execute(self.code, self.unif.addresses()[0], timeout_sec=100, thread=self.num_qpus)   




@qpu
def kernel(asm, num_qpus):
    A_ADDR=0
    A_STR=1
    B_ADDR=2
    B_STR=3
    C_ADDR=4
    HBLOCK=5
    WBLOCK=6
    LOOP_I=7
    LOOP_J=8
    LOOP_K=9
    FRAC_W=10
    FRAC_H=11
    A_SIZE=12
    B_SIZE=13
    C_SIZE=14
    LOOP_CH=15
    
    eidx(r0).mov(r2, 0)
    for idx in [A_ADDR, A_STR, B_ADDR, B_STR, C_ADDR, HBLOCK, WBLOCK, LOOP_I, LOOP_J, LOOP_K, FRAC_W, FRAC_H, A_SIZE, B_SIZE, C_SIZE, LOOP_CH]:
        nop(sig=ldunifrf(r5))
        sub(null, r0, idx, cond='pushz')
        mov(r2, r5, cond='ifa')

    cidx=rf49
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
    umul24(r3,r5,r1)

    #端数処理
    rotate(broadcast,r2,-FRAC_W)
    sub(null,r1,3,cond='pushz')
    sub(r3,r3,r5, cond='ifa')
    mov(r1,r3)

    sub(null, r4, B_ADDR, cond='pushz')
    add(r2, r2, r1, cond='ifa')    




    #numqpu%4
    #A set

    shr(r3,r0,2)
    rotate(broadcast,r2,-HBLOCK)
    mov(r0,r5)
    rotate(broadcast,r2,-FRAC_H)
    sub(r0,r0,r5)

    umul24(r3,r0,r3)
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


    mov(cidx,0)
    with loop as cloop:

        

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
            umul24(r0,ldi16,iidx)

            #端数処理 
            # if HBLOCK - i * 16 < 0:
            #     i - (16 + (HBLOCK - i*16))        
            add(r1,r0,ldi16)
            rotate(broadcast,r2,-HBLOCK)        
            sub(r1,r5,r1,cond = 'pushn')
            b(R.fraction_i_end,cond='anyna')
            mov(r1,0) #nop
            eidx(a_cur) #nop
            rotate(broadcast,r2,-A_STR) #nop

            add(r0,r0,r1)
            eidx(a_cur)
            rotate(broadcast,r2,-A_STR)

            L.fraction_i_end

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

                # if WBLOCK - j * 32 * 4(bytes) < 0:
                #     r0 - (128 + (WBLOCK - j * 128)
                add(r3,r0,ldi128)
                rotate(broadcast,r2,-WBLOCK)
                sub(r3,r5,r3,cond = 'pushn')
                b(R.fraction_j_end,cond='anyna')
                mov(kidx,0)
                eidx(b_cur)
                mov(rf48,0)


                add(r0,r0,r3)
                mov(rf48,r3)
                mov(kidx,0)
                eidx(b_cur)

                L.fraction_j_end

                #2 : eidx x 4 + B_ADDR
                shl(b_cur,b_cur,2)
                rotate(broadcast,r2,-B_ADDR)
                add(b_cur,b_cur,r5)

                #1 + 2
                add(b_cur,b_cur,r0)

                with loop as kloop:
                    mov(tmua,a_cur,sig=thrsw)
                    add(a_cur,a_cur,4) #nop()
                    add(kidx,kidx,1) #nop()
                    nop(sig=ldtmu(r4))
                    for lj in range(2):
                        stp = lj*16
                        mov(tmua,b_cur,sig=thrsw)
                        if lj==0:
                            add(b_cur,b_cur,simd_stp) #nop()
                        else:
                            nop()
                        nop()
                        nop(sig=ldtmu(r3))
                        rotate(broadcast,r4,0)
                        fmul(r0,r5,r3)                    
                        for li in range(15):
                            rotate(broadcast,r4,-(li+1))
                            fadd(rf[stp+li],rf[stp+li],r0).fmul(r0,r5,r3)
                        fadd(rf[stp+15],rf[stp+15],r0)
                    rotate(broadcast,r2,-LOOP_K)
                    sub(null,r5,kidx,cond='pushz')
                    kloop.b(cond='anyna')
                    sub(b_cur,b_cur,simd_stp) #nop()
                    rotate(broadcast,r2,-B_STR) #nop()
                    add(b_cur,b_cur,r5) #nop() 


                umul24(r0,ldi16,iidx)
                rotate(broadcast,r2,-B_STR)
                umul24(r0,r5,r0)

                eidx(c_cur)
                umul24(c_cur,c_cur,4)

                umul24(r1,r1,r5) # 端数処理

                add(c_cur,c_cur,r0)

                # 32 x 4(float) x jidx                
                umul24(r0,ldi128,jidx)
                add(r0,r0,rf48)
                rotate(broadcast,r2,-C_ADDR)
                add(c_cur,c_cur,r5)
                add(c_cur,c_cur,r0)
                add(c_cur,c_cur,r1) #端数処理

                rotate(broadcast,r2,-B_STR)
                sub(r0,r5,simd_stp)
                for li in range(16):
                    mov(tmud,rf[li])
                    mov(tmua,c_cur)
                    add(c_cur,c_cur,simd_stp)
                    mov(rf[li],0.0)
                    tmuwt()
                    mov(tmud,rf[li + 16])
                    mov(tmua,c_cur)
                    add(c_cur,c_cur,r0)
                    mov(rf[li+16],0.0)
                    tmuwt()

                rotate(broadcast,r2,-LOOP_J)
                add(jidx,jidx,1)
                sub(null,r5,jidx,cond = 'pushz')
                jloop.b(cond='anyna')
                rotate(broadcast,r2,-A_STR) #nop()
                sub(a_cur,a_cur,r5) #nop()
                nop()

            add(iidx,iidx,1)
            rotate(broadcast,r2,-LOOP_I)
            sub(null,r5,iidx,cond = 'pushz')
            iloop.b(cond='anyna')
            nop()
            nop()
            nop()

            
        eidx(r0)
        rotate(broadcast,r2,-A_SIZE)
        sub(null, r0, A_ADDR, cond='pushz')
        add(r2, r2, r5, cond='ifa')

        rotate(broadcast,r2,-B_SIZE)
        sub(null, r0, B_ADDR, cond='pushz')
        add(r2, r2, r5, cond='ifa')

        rotate(broadcast,r2,-C_SIZE)
        sub(null, r0, C_ADDR, cond='pushz')
        add(r2, r2, r5, cond='ifa')
        
        add(cidx,cidx,1)
        rotate(broadcast,r2,-LOOP_CH)
        sub(null,r5,cidx,cond = 'pushz')
        cloop.b(cond='anyna')
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
