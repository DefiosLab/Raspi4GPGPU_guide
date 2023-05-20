#####################################################################
# 2のn乗を計算する
# GPUで整数値は24bitで扱われるので2^24が限界
# 参考：
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
    g['reg_Out_stride'] = g['rf3']

    g['reg_In_cur']     = g['rf4']
    g['reg_Out_cur']    = g['rf5']

    # uniformから値を取り出す
    # uniformの読み取り位置はインクリメントされる(pop的動作)
    nop(sig=ldunifrf(reg_In_base))
    nop(sig=ldunifrf(reg_In_stride))
    nop(sig=ldunifrf(reg_Out_base))
    nop(sig=ldunifrf(reg_Out_stride))

    # TMU用 Baseアドレス生成
    eidx(r0)        # r0 = [0 ... 15]
    shl(r0, r0, 2)  # 各数値を4倍(float32のバイト数分)
    add(reg_In_cur,  reg_In_base,  r0) # Baseアドレスから ストライド=4バイトのアドレスベクトルを生成
    add(reg_Out_cur, reg_Out_base, r0) # Baseアドレスから ストライド=4バイトのアドレスベクトルを生成

    # データの読み込み
    mov(tmua, reg_In_cur, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(r0))

    # 初期値
    mov(r1, 1)

    # 2の0乗の場合ループをスキップする
    mov(null, r0, cond = 'pushz')
    b(R.end, cond = 'alla')
    nop(); nop(); nop()


    # 2のn乗(0<n)の場合
    with loop as li:
      # r1 = 1*2*2*2*...
      umul24(r1, r1, 2)
      
      sub(r0, r0, 1, cond = 'pushz')
      li.b(cond='allna')
      nop() # delay slot
      nop() # delay slot
      nop() # delay slot


    L.end

    mov(tmud, r1)           # 書き出すデータ
    mov(tmua, reg_Out_cur)  # 書き出し先アドレスベクトル


    # GPUコードを終了する
    exit_qpu()

def main():
    with Driver() as drv:
        # params setting
        inp = drv.alloc((1, 16), dtype='uint32')
        out = drv.alloc((1, 16), dtype='uint32')

        print("整数を入力")
        inp[0][:] = input()

        out[0][:] = 0
        
        # uniform setting
        unif = drv.alloc(4, dtype='uint32')
        unif[0] = inp.addresses()[0,0]
        unif[1] = inp.strides[0]
        unif[2] = out.addresses()[0,0]
        unif[3] = out.strides[0]

        # Run the program
        code = drv.program(kernel)
        drv.execute(code, unif.addresses()[0], thread=1)


        print(' input '.center(80, '='))
        print(inp[0])
        print(' 2^n '.center(80, '='))
        print(out[0])
        print(' error '.center(80, '='))
        print(np.abs(2**inp[0] - out[0]))

if __name__ == '__main__':
    main()