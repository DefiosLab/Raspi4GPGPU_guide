#####################################################################
# 
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
def output_test(asm):
    # uniformの何番目になんの値があるか
    ADDR0 = 0
    ADDR1 = 1
    ADDR2 = 2
    ADDR3 = 3

    # element_number
    eidx(r2)         # r0 = [0 ... 15]
    mov(r0, 0)

    # uniformから値を取り出す
    # uniformの読み取り位置はインクリメントされる(pop的動作)
    nop(sig=ldunifrf(rf0))
    sub(null, r2, ADDR0, cond='pushz') 
    mov(r0, rf0, cond='ifa')

    nop(sig=ldunifrf(rf0))
    sub(null, r2, ADDR1, cond='pushz') 
    mov(r0, rf0, cond='ifa')

    nop(sig=ldunifrf(rf0))
    sub(null, r2, ADDR2, cond='pushz') 
    mov(r0, rf0, cond='ifa')

    nop(sig=ldunifrf(rf0))
    sub(null, r2, ADDR3, cond='pushz') 
    mov(r0, rf0, cond='ifa')

    # アドレスを取り出す
    nop()
    rotate(broadcast, r0, -ADDR0)
    mov(r1, r5)

    # element_number
    eidx(r2)         # r2 = [0 ... 15]
    shl(r2, r2, 2)   # 各数値を4倍
    add(r1, r1, r2)  # result[] のアドレスから ストライド=4バイトのアドレスベクトルを生成

    mov(tmud, r0)  # 書き出すデータ
    mov(tmua, r1)  # 書き出し先アドレスベクトル
    tmuwt()

    # GPUコードを終了する
    exit_qpu()


def main():
    with Driver() as drv:
        code = drv.program(output_test)

        result = drv.alloc(16, dtype='uint32')
        result[:] = 0

        unif = drv.alloc(4, dtype='uint32')
        unif[0] = result.addresses()[0]
        unif[1] = 1
        unif[2] = 2
        unif[3] = 3

        print("before")
        print(result)
        num_qpus = 1
        drv.execute(code, unif.addresses()[0], thread=num_qpus)
        print("after")
        print(result)

if __name__ == '__main__':
    main()

