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

    # element_index
    eidx(r2)         # r2 = [0 ... 15]
    shl(r2, r2, 2)   # 各数値を4倍
    add(r0, r0, r2)  # result[] のアドレスから ストライド=4バイトのアドレスベクトルを生成

    # r1を7で埋める
    mov(r1, 7)

    mov(tmud, r1)  # 書き出すデータ
    mov(tmua, r0)  # 書き出し先アドレスベクトル
    tmuwt()

    # GPUコードを終了する
    exit_qpu()

def main():
    with Driver() as drv:
        out = drv.alloc(16, 'uint32')
        out[:] = 0.0

        unif = drv.alloc(1, dtype='uint32')
        unif[0] = out.addresses()[0]

        print(' out_Before '.center(80, '='))
        print(out)

        num_qpus = 1
        code = drv.program(kernel)
        drv.execute(code, unif.addresses()[0], thread=num_qpus)

        print(' out_After '.center(80, '='))
        print(out)

if __name__ == '__main__':
    main()
