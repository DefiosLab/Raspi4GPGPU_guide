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
        self.copy_flg1=False
        self.copy_flg2=False
        self.num_qpus=8
        self.simd_width=16
        num_thy=2
        num_thx=4        
        self.input_data1 = self.tensor_data[self.input_name[0]] 
        self.input_data2 = self.tensor_data[self.input_name[1]]
        self.output_data = self.tensor_data[self.output_name]
        data1_shape=list(self.input_data1.shape)
        data2_shape=list(self.input_data2.shape)
        output_shape=list(self.output_data.shape)
        self.p,self.q = data1_shape[-2:]
        self.q,self.r = data2_shape[-2:]
        self.p_mod = self.p % (num_thy*16)
        self.r_mod = self.r % (num_thx*32)
        hblock=self.p/num_thy
        wblock=self.r/num_thx
        j_idx=wblock/32
        i_idx=hblock/16
        k_idx=self.q
        if self.p_mod > 0:
            new_p = math.ceil(self.p / (num_thy*16)) * (num_thy*16)
            data1_shape[-2] = int(new_p)
            output_shape[-2] = int(new_p)
            self.input_data1 = self.drv.alloc(data1_shape,dtype=self.input_data1.dtype)
            if self.input_name[0] in self.init_names:
                self.input_data1[...,:self.p,:self.q]=self.tensor_data[self.input_name[0]]
                self.tensor_data[self.input_name[0]]=self.input_data1
            else:
                self.copy_flg1=True
            hblock=new_p/num_thy
            i_idx=hblock/16
        else:        
            if self.input_name[0] in self.init_names:
                cpu_w = self.tensor_data[self.input_name[0]]
                self.tensor_data[self.input_name[0]] = self.drv.alloc(cpu_w.shape,cpu_w.dtype)
                self.tensor_data[self.input_name[0]][:]=cpu_w
                self.input_data1 = self.tensor_data[self.input_name[0]]
                
        if self.r_mod > 0:
            new_r = math.ceil(self.r / (num_thx*32)) * (num_thx*32)
            data2_shape[-1] = int(new_r)
            output_shape[-1] = int(new_r)
            self.input_data2 = self.drv.alloc(data2_shape,dtype=self.input_data2.dtype)      
            if self.input_name[1] in self.init_names:
                self.input_data2[...,:self.q,:self.r]=self.tensor_data[self.input_name[1]]
                self.tensor_data[self.input_name[1]]=self.input_data2
            else:
                self.copy_flg2=True
            wblock=new_r/num_thx
            j_idx=wblock/32
        else:        
            if self.input_name[1] in self.init_names:
                cpu_w = self.tensor_data[self.input_name[1]]
                self.tensor_data[self.input_name[1]] = self.drv.alloc(cpu_w.shape,cpu_w.dtype)
                self.tensor_data[self.input_name[1]][:]=cpu_w
                self.input_data2 = self.tensor_data[self.input_name[1]]
        if self.r_mod > 0 or self.p_mod >0:
            self.output_data = self.drv.alloc(output_shape,dtype=self.output_data.dtype)
        self.output_data[:]=0
        self.unif = self.drv.alloc(16, dtype='uint32')
        self.unif[0] = self.input_data1.address
        self.unif[1] = self.input_data1.strides[-2]  
        self.unif[2] = self.input_data2.address
        self.unif[3] = self.input_data2.strides[-2]
        self.unif[4] = self.output_data.address
        self.unif[5] = hblock
        self.unif[6] = wblock*4 #float
        self.unif[7] = i_idx
        self.unif[8] = j_idx
        self.unif[9] = k_idx
        self.code = self.drv.program(kernel, num_qpus=self.num_qpus)        
            
    def run(self):
        # if len(self.input_data1.shape)>=3:
        #     if self.input_data1.shape[-3] > 1:
        #         input_data1 = self.tensor_data[self.input_name[0]] 
        #         input_data2 = self.tensor_data[self.input_name[1]] 
        #         output = input_data1 @ input_data2
        #         if self.use_gpu:
        #             self.tensor_data[self.output_name][:]=output
        #             return
        #         else:
        #             self.tensor_data[self.output_name]=output
        #             return 
        if self.copy_flg1:
            self.input_data1[...,:self.p,:self.q]=self.tensor_data[self.input_name[0]]
        if self.copy_flg2:
            self.input_data2[...,:self.q,:self.r]=self.tensor_data[self.input_name[1]]
        print(self.input_data1.shape,self.input_data2.shape)
        print(self.unif)
        # self.input_data1[:]=1
        # self.input_data2[:]=1
        self.drv.execute(self.code, self.unif.addresses()[0], thread=self.num_qpus)
        cpu = self.input_data1@self.input_data2
        gpu = self.output_data
        print(cpu)
        print(gpu)
        print('maximum relative error : {:.4e}'.format(float(np.max(np.abs(cpu-gpu)/cpu))))        
        # print(self.output_data)
        # print(self.input_data1@self.input_data2)
        if self.p_mod > 0 or self.r_mod > 0:
            self.tensor_data[self.output_name]=self.output_data[...,:self.p,:self.r]
        print(self.input_data1.dtype,self.input_data2.dtype,self.output_data.dtype)



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



