#####################################################################
# 何もしない，最もシンプルなプログラム
#####################################################################
import numpy as np

from videocore6.assembler import qpu
from videocore6.driver import Driver

@qpu
def basic_temp(asm):
    ### ここにカーネルプログラムを書く ###

    nop() # なにもしない

    # GPUコードの終了を明示
    exit()

def main():
    with Driver() as drv:
        # ここにホストプログラムを書く

        # 実行するカーネルプログラム
        code = drv.program(basic_temp)

        # カーネルプログラム実行
        num_qpus = 1  # スレッド数(1~8)
        drv.execute(code, thread=num_qpus)

        '''
        note:オプションとデフォルト値
        drv.execute(self, code, uniforms=None, timeout_sec=10, workgroup=(16, 1, 1), wgs_per_sg=16, thread=1)
        '''

if __name__ == '__main__':
    main()
