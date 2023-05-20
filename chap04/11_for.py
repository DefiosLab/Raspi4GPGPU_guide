#####################################################################
# ベクトル同士の和を求める
# 参考：https://github.com/nineties/py-videocore/blob/master/examples/hello_world.py
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
    # uniformから値を取り出す
    # uniformの読み取り位置はインクリメントされる(pop的動作)
    nop(sig=ldunifrf(r1))
    nop(sig=ldunifrf(r2))

    # element_number
    eidx(r3)         # r2 = [0 ... 15]
    shl(r3, r3, 2)   # 各数値を4倍
    add(r1, r1, r3)  # result[] のアドレスから ストライド=4バイトのアドレスベクトルを生成
    add(r2, r2, r3)  # result[] のアドレスから ストライド=4バイトのアドレスベクトルを生成

    mov(tmua, r1, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(r0))

    for i in range(5):
        fadd(r0, r0, 1.0)

    mov(tmud, r0)  # 書き出すデータ
    mov(tmua, r2)  # 書き出し先アドレスベクトル
    tmuwt()
    
    # GPUコードを終了する
    exit_qpu()


def main():
    with Driver() as drv:
        # params setting
        list_a = drv.alloc(16, dtype='float32')
        out = drv.alloc(16, dtype='float32')

        list_a[:] = 0.0
        
        # uniform setting
        unif = drv.alloc(3, dtype='uint32')
        unif[0] = list_a.addresses()[0]
        unif[1] = out.addresses()[0]

        # Run the program
        code = drv.program(kernel)
        drv.execute(code, unif.addresses()[0], thread=1)

        print(' out '.center(80, '='))
        print(out)

        for i in range(5):
            list_a += 1.0
            cpu_ans = list_a

        error   = cpu_ans - out
        print(' error '.center(80, '='))
        print(np.abs(error))

if __name__ == '__main__':
    main()
