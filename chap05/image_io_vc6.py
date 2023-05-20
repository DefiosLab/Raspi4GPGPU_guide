#####################################################################
# 
# 参考：https://github.com/Taiki-azrs/RaspiGPGPU_guide/tree/master/chap05
#####################################################################
#coding:utf-8
from time import clock_gettime, CLOCK_MONOTONIC
import numpy as np
from PIL import Image, ImageFilter

from videocore6.assembler import qpu
from videocore6.driver import Driver


def getsec():
    return clock_gettime(CLOCK_MONOTONIC)

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
def kernel(asm, num_qpus):
    g = globals()
    g['reg_In_base']    = g['rf0']
    g['reg_In_stride']  = g['rf1']
    g['reg_Out_base']   = g['rf2']
    g['reg_Out_stride'] = g['rf3']
    g['reg_H']          = g['rf4']
    g['reg_W']          = g['rf5']

    g['reg_In_cur']     = g['rf6']
    g['reg_Out_cur']    = g['rf7']
    g['reg_In']         = g['rf8']
    g['reg_Out']        = g['rf9']
    g['reg_loop_h']     = g['rf10']
    g['reg_loop_w']     = g['rf11']

    g['reg_qpu_num']     = g['rf12']

    # uniformから値を取り出す
    # uniformの読み取り位置はインクリメントされる(pop的動作)
    nop(sig=ldunifrf(reg_In_base))
    nop(sig=ldunifrf(reg_In_stride))
    nop(sig=ldunifrf(reg_Out_base))
    nop(sig=ldunifrf(reg_Out_stride))
    nop(sig=ldunifrf(reg_H))
    nop(sig=ldunifrf(reg_W))

    if num_qpus == 1:
        mov(reg_qpu_num, 0)
    elif num_qpus == 8:
        tidx(r0)
        shr(r0, r0, 2)
        band(reg_qpu_num, r0, 0b1111)
    else:
        raise Exception('num_qpus must be 1 or 8')


    # TMU用 Baseアドレス生成
    eidx(r0)        # r0 = [0 ... 15]
    shl(r0, r0, 2)  # 各数値を4倍(float32のバイト数分)
    add(reg_In_cur,  reg_In_base,  r0) # Baseアドレスから ストライド=4バイトのアドレスベクトルを生成
    add(reg_Out_cur, reg_Out_base, r0) # Baseアドレスから ストライド=4バイトのアドレスベクトルを生成

    # TMU用 Strideアドレス生成
    # stride幅 = 1要素のバイト幅(今回はfloat32なので4バイト) * 1度に何個ずつアクセスするか
    # float32を16個ずつ読み書きする場合，アドレスの移動幅は64byte
    shl(reg_In_stride, reg_In_stride, 4)
    shl(reg_Out_stride, reg_Out_stride, 4)

    
    ##################################
    # マルチコア動作のためのアドレス計算
    ##################################
    # 1ブロックのサイズ = 1回のIOのバイト幅(64byte) * width * 1qpuが担当するheight
    umul24(r0, reg_In_stride, reg_W)
    umul24(r0, r0, reg_H)

    # baseからのアドレス移動幅 = 1ブロックのサイズ * qpu_num
    umul24(r0, r0, reg_qpu_num)

    # 自分が担当すべきブロックの先頭アドレスまでアドレスを移動
    add(reg_In_cur, reg_In_cur, r0)
    add(reg_Out_cur, reg_Out_cur, r0)


    ##################################
    # ループ開始
    ##################################
    mov(reg_loop_h, reg_H)
    with loop as lh:

      mov(reg_loop_w, reg_W)
      with loop as lw:
        # データの読み込み
        mov(tmua, reg_In_cur, sig = thrsw)
        nop()
        nop()
        nop(sig = ldtmu(reg_In))

        mov(tmud, reg_In)       # 書き出すデータ
        mov(tmua, reg_Out_cur)  # 書き出し先アドレスベクトル

        # addressのインクリメント
        add(reg_In_cur,  reg_In_cur,  reg_In_stride)
        add(reg_Out_cur, reg_Out_cur, reg_Out_stride)

        sub(reg_loop_w, reg_loop_w, 1, cond = 'pushz')
        lw.b(cond='anyna')
        nop() # delay slot
        nop() # delay slot
        nop() # delay slot
    
      sub(reg_loop_h, reg_loop_h, 1, cond = 'pushz')
      lh.b(cond='anyna')
      nop() # delay slot
      nop() # delay slot
      nop() # delay slot

    # This synchronization is needed between the last TMU operation and the
    # program end with the thread switch just before the loop above.
    barrierid(syncb, sig=thrsw)
    nop()
    nop()

    # GPUコードを終了する
    exit_qpu()


def main():
    # set 1 or 8
    num_qpus = 8

    # 画像サイズ
    H=360
    W=320

    pil_img = Image.open('./LLL.png').convert('L')

    with Driver() as drv:
        # params setting
        inp = drv.alloc((H, W), dtype='float32')
        out = drv.alloc((H, W), dtype='float32')

        inp[:] = np.asarray(pil_img)
        out[:] = 0
        
        # uniform setting
        unif = drv.alloc(6, dtype='uint32')
        unif[0] = inp.addresses()[0,0]
        unif[1] = inp.strides[1]  # Wが1つインクリメントされたらポインタが何バイト移動するか(横1要素のサイズ)
        unif[2] = out.addresses()[0,0]
        unif[3] = out.strides[1]
        unif[4] = H/num_qpus
        unif[5] = W/16

        code = drv.program(kernel, num_qpus=num_qpus)

        # Run the program
        start = getsec()
        drv.execute(code, unif.addresses()[0], thread=num_qpus)
        time_gpu = (getsec() - start)*1000.0

        print(f'QPU:   {time_gpu:.4} msec')
        print(' diff '.center(80, '='))
        print(np.abs(inp - out))

        pil_img = Image.fromarray(out.astype(np.uint8))
        pil_img.save('./out.png')

if __name__ == '__main__':
    main()