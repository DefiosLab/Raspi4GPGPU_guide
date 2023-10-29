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
        if self.copy_flg1:
            self.input_data1[...,:self.p,:self.q]=self.tensor_data[self.input_name[0]]
        if self.copy_flg2:
            self.input_data2[...,:self.q,:self.r]=self.tensor_data[self.input_name[1]]        

        inp1_shape = list(self.input_data1.shape)
        inp2_shape = list(self.input_data2.shape)
        if len(inp1_shape) <3:
            inp1_shape.insert(0,1)
        if len(inp2_shape) <3:
            inp2_shape.insert(0,1)            
        inp1_range = range(inp1_shape[-3])
        inp2_range = range(inp2_shape[-3])
        max_len = max(len(inp1_range),len(inp2_range))
        if max_len > len(inp1_range):
            inp1_range=[0]*max_len
        if max_len > len(inp2_range):
            inp2_range=[0]*max_len
        offset1=len(inp1_shape)*[0]
        offset2=len(inp2_shape)*[0]
        offset3=max(len(inp2_shape),len(inp1_shape))*[0]                
        for i,j in zip(inp1_range,inp2_range):
            offset1[-3]=i
            offset2[-3]=j
            offset3[-3]=max(i,j)
            if len(self.input_data1.shape) > 2 :
                self.unif[0] = self.input_data1.addresses()[tuple(offset1)]
            if len(self.input_data2.shape) > 2:
                self.unif[2] = self.input_data2.addresses()[tuple(offset2)]
            if len(self.output_data.shape) > 2:                
                self.unif[4] = self.output_data.addresses()[tuple(offset3)]
            self.drv.execute(self.code, self.unif.addresses()[0], thread=self.num_qpus)
            
        else:
            self.drv.execute(self.code, self.unif.addresses()[0], thread=self.num_qpus)        

        gpu = self.output_data
        # cpu = self.input_data1@self.input_data2
        # print('maximum relative error : {:.4e}'.format(float(np.max(np.abs(cpu-gpu)/np.maximum(np.abs(cpu),1e-10)))))        
        self.tensor_data[self.output_name][:] = gpu[...,:self.p,:self.r]




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
