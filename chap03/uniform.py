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
    nop(sig=ldunifrf(r0))  # r0 = [2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
    nop(sig=ldunifrf(r1))  # r1 = [3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3]
    nop(sig=ldunifrf(r3))  # r3 = [out.addresses()[0] x 16]

    add(r0, r0, r1)

    mov(tmud, r0)  # 書き出すデータ
    mov(tmua, r3)  # 書き出し先アドレスベクトル
    tmuwt()
    
    # GPUコードを終了する
    exit_qpu()


def main():
    with Driver() as drv:
        out = drv.alloc(16, dtype='uint32')
        
        # uniform setting
        unif = drv.alloc(3, dtype='uint32')
        unif[0] = 2
        unif[1] = 3
        unif[2] = out.addresses()[0]

        # Run the program
        code = drv.program(kernel)
        drv.execute(code, uniforms=unif.addresses()[0], thread=1)

        print(' uniform[0] '.center(80, '='))
        print(unif[0])
        print(' uniform[1] '.center(80, '='))
        print(unif[1])
        print(' uniform[0] + uniform[1] '.center(80, '='))
        print(out)

if __name__ == '__main__':
    main()
