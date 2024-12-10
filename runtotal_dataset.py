# 这个文件是一个辅助文件，用于并行处理数据集，运行 python runtotal_dataset.py [file_name] 即可并行运行 file_name 文件
import json
import math
import time
import subprocess
import os
import sys 
import pickle

# 主函数入口
if __name__ == '__main__':
    # 从命令行参数获取文件名
    file_name = sys.argv[1]

    # 定义原始数据路径
    total_raw_data_path = 'RAW_DATA'
    
    # 加载原始数据
    all_raw_base, all_raw_a, all_raw_b, all_raw_res = json.load(open('%s/raw_data' % (total_raw_data_path)))
    
    # 获取原始数据的总数
    all_num = len(all_raw_base)
    
    # 存储子进程的列表
    jobs = []
    
    # 最大同时运行的进程数
    max_num = 10
    
    # 每个子进程处理的数据量
    each_num = 1000
    
    # 根据数据总数和每个子进程处理的数据量，计算需要多少个子进程
    for i in range(math.ceil(all_num / each_num)):
        while True:
            run_num = 0 
            # 统计当前正在运行的子进程数
            for x in jobs:
                if x.poll() is None:
                    run_num += 1
            # 如果当前运行的子进程数小于最大允许的进程数，则跳出循环
            if run_num < max_num:
                break
            # 如果当前运行的子进程数达到最大值，则等待1秒后再检查
            time.sleep(1)
        
        # 计算当前子进程处理的数据范围
        start = i * each_num
        end = min((i + 1) * each_num, all_num)
        
        # 启动一个新的子进程来处理数据
        p = subprocess.Popen("python %s %s %s" % (file_name, start, end), shell=True)
        
        # 将子进程对象添加到列表中
        jobs.append(p)
        
        # 等待1秒后再启动下一个子进程
        time.sleep(1)
    
    # 等待所有子进程完成
    for job in jobs:
        job.wait()