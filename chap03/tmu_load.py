#####################################################################
# 16個のラッキー7をQPUからHostに出力するサンプルプログラム
# TMUを使った[レジスタ]->[TMU]->[GPUメモリ]転送が確認できる
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
    nop(sig=ldunifrf(r0))
    nop(sig=ldunifrf(r1))

    # element_index
    eidx(r2)         # r2 = [0 ... 15]
    shl(r2, r2, 2)   # 各数値を4倍
    add(r0, r0, r2)  # inp[] のアドレスから ストライド=4バイトのアドレスベクトルを生成
    add(r1, r1, r2)  # out[] のアドレスから ストライド=4バイトのアドレスベクトルを生成

    mov(tmua, r0, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(r3))

    mov(tmud, r3)  # 書き出すデータ
    mov(tmua, r1)  # 書き出し先アドレスベクトル
    tmuwt()

    # GPUコードを終了する
    exit_qpu()


def main():
    with Driver() as drv:
        list_a = drv.alloc(16, dtype='float32')
        list_a[:] = np.arange(1, 17)

        out = drv.alloc(16, dtype='float32')
        out[:] = 0.0

        unif = drv.alloc(2, dtype='uint32')
        unif[0] = list_a.addresses()[0]
        unif[1] = out.addresses()[0]

        print(' list_a '.center(80, '='))
        print(list_a)
        print(' out_Before '.center(80, '='))
        print(out)
        
        num_qpus = 1
        code = drv.program(kernel)
        drv.execute(code, unif.addresses()[0], thread=num_qpus)
        
        print(' out_After '.center(80, '='))
        print(out)

if __name__ == '__main__':
    main()
