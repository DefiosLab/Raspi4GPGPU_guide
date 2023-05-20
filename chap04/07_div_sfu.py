#####################################################################
# Floatベクトル同士の除算をする
# 参考：https://github.com/Taiki-azrs/RaspiGPGPU_guide/blob/master/chap04/02_fmul.py
#####################################################################
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
    g['reg_In_base']    = g['rf0']
    g['reg_In_stride']  = g['rf1']
    g['reg_Out_base']   = g['rf2']

    g['reg_A']          = g['rf3']
    g['reg_B']          = g['rf4']

    # uniformから値を取り出す
    # uniformの読み取り位置はインクリメントされる(pop的動作)
    nop(sig=ldunifrf(reg_In_base))
    nop(sig=ldunifrf(reg_In_stride))
    nop(sig=ldunifrf(reg_Out_base))

    # element_number
    eidx(r0)         # r2 = [0 ... 15]
    shl(r0, r0, 2)   # 各数値を4倍
    add(reg_In_base,  reg_In_base,  r0)  # result[] のアドレスから ストライド=4バイトのアドレスベクトルを生成
    add(reg_Out_base, reg_Out_base, r0)  # result[] のアドレスから ストライド=4バイトのアドレスベクトルを生成

    # A_refを読む
    mov(tmua, reg_In_base, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(reg_A))

    # B_refを読む
    add(reg_In_base, reg_In_base, reg_In_stride)
    mov(tmua, reg_In_base, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(reg_B))

    recip(reg_B, reg_B)        # 割る数の逆数を求める
    fmul(reg_A, reg_A, reg_B)  # r2 = r2 * recip(reg_B)

    mov(tmud, reg_A)         # 書き出すデータ
    mov(tmua, reg_Out_base)  # 書き出し先アドレスベクトル
    tmuwt()
    
    # GPUコードを終了する
    exit_qpu()


def main():
    with Driver() as drv:
        # Input vectors
        A_ref = np.random.random(16).astype('float32')
        B_ref = np.random.random(16).astype('float32')

        # params setting
        inp = drv.alloc((2, 16), dtype='float32')
        out = drv.alloc(16, dtype='float32')

        inp[0][:] = A_ref
        inp[1][:] = B_ref
        
        # uniform setting
        unif = drv.alloc(3, dtype='uint32')
        unif[0] = inp.addresses()[0,0]
        unif[1] = inp.strides[0]
        unif[2] = out.addresses()[0]

        # Run the program
        code = drv.program(kernel)
        drv.execute(code, unif.addresses()[0], thread=1)


        print(' a '.center(80, '='))
        print(A_ref)
        print(' b '.center(80, '='))
        print(B_ref)
        print(' a/b '.center(80, '='))
        print(out)
        print(' error '.center(80, '='))
        print(np.abs((A_ref / B_ref) - out))

if __name__ == '__main__':
    main()