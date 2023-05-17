
import subprocess


def alter(file, old_str, new_str):
    """
    替换文件中的字符串
    :param file:文件名
    :param old_str:就字符串
    :param new_str:新字符串
    :return:
    """
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if old_str in line:
                line = line.replace(old_str, new_str)
            file_data += line
    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)

manu_seeds = [f"manu_seed={i}" for i in range(5)]
txtpaths = ["981762","981808","981814"]

for i,j in zip(txtpaths,txtpaths[1:]+txtpaths[0:1]):
    for m,n in zip(manu_seeds,manu_seeds[1:]+manu_seeds[0:1]):
        print(f"正在运行{i}{m}","*"*10)
        subprocess.run(["python","eval.py"],check=True,shell=True)
        print(f"{i}{m}运行完成","*"*10)
        alter("config.py",m,n)
    alter("config.py", i, j)
