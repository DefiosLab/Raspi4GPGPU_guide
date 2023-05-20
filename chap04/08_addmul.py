#####################################################################
# 
# 
#####################################################################
#coding:utf-8
import numpy as np

from videocore6.assembler import qpu
from videocore6.driver import Driver

def exit_qpu():
    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()

@qpu
def kernel(asm):
    g = globals()
    g['reg_InA_base']   = g['rf0']
    g['reg_InB_base']   = g['rf1']
    g['reg_OutA_base']  = g['rf2']
    g['reg_OutB_base']  = g['rf3']

    # uniformから値を取り出す
    # uniformの読み取り位置はインクリメントされる(pop的動作)
    nop(sig=ldunifrf(reg_InA_base))
    nop(sig=ldunifrf(reg_InB_base))
    nop(sig=ldunifrf(reg_OutA_base))
    nop(sig=ldunifrf(reg_OutB_base))

    eidx(r0)        # r0 = [0 ... 15]
    shl(r0, r0, 2)  # 各数値を4倍(float32のバイト数分)
    add(reg_InA_base,  reg_InA_base,  r0) # Baseアドレスから ストライド=4バイトのアドレスベクトルを生成
    add(reg_InB_base,  reg_InB_base,  r0)
    add(reg_OutA_base, reg_OutA_base, r0)
    add(reg_OutB_base, reg_OutB_base, r0)

    # データの読み込み
    mov(tmua, reg_InA_base, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(rf10))

    mov(tmua, reg_InB_base, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(rf11))

    fadd(r2, rf10, rf11).fmul(r3, rf10, rf11)

    # データの書き出し
    mov(tmud, r2)
    mov(tmua, reg_OutA_base)
    tmuwt()

    mov(tmud, r3)
    mov(tmua, reg_OutB_base)
    tmuwt()

    # GPUコードを終了する
    exit_qpu()


def main():
    with Driver() as drv:
        # Input vectors
        list_a = drv.alloc(16, dtype='float32')
        list_a[:] = 3.0

        list_b = drv.alloc(16, dtype='float32')
        list_b[:] = 4.0

        # Output vectors
        out_a = drv.alloc(16, dtype='float32')
        out_a[:] = 0
        out_b = drv.alloc(16, dtype='float32')
        out_b[:] = 0
        
        # uniform setting
        unif = drv.alloc(4, dtype='uint32')
        unif[0] = list_a.addresses()[0]
        unif[1] = list_b.addresses()[0]
        unif[2] = out_a.addresses()[0]
        unif[3] = out_b.addresses()[0]

        # Run the program
        code = drv.program(kernel)
        drv.execute(code, unif.addresses()[0], thread=1)

        print(' list_a, list_b '.center(80, '='))
        print(list_a)
        print(list_b)
        print(' out_a, out_b '.center(80, '='))
        print("add =", out_a)
        print("mul =", out_b)

if __name__ == '__main__':
    main()
