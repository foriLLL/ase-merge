# runtotal_dataset.py
import json
import math
import time
import subprocess
import os
import sys

def aggregate_results(aggregated, new_result):
    """
    根据需要实现聚合逻辑。
    这里以简单的计数相加为例。
    """
    aggregated['both_overflows'] += new_result['both_overflows']
    aggregated['only_input_overflows'] += new_result['only_input_overflows']
    aggregated['only_res_overflows'] += new_result['only_res_overflows']
    aggregated['data_num'] += new_result['data_num']
    return aggregated

def main(file_name):
    # 定义原始数据路径
    total_raw_data_path = 'RAW_DATA'
    
    # 加载原始数据
    raw_data_file = os.path.join(total_raw_data_path, 'graphQL_raw_data_sample_20')
    with open(raw_data_file, 'r') as f:
        all_raw_base, all_raw_a, all_raw_b, all_raw_res = json.load(f)
    
    # 获取原始数据的总数
    all_num = len(all_raw_base)
    
    # 存储子进程的列表
    jobs = []
    
    # 最大同时运行的进程数
    max_num = 10
    
    # 每个子进程处理的数据量
    each_num = 1000
    
    # 初始化聚合结果
    aggregated_result = {
        "both_overflows": 0,
        "only_input_overflows": 0,
        "only_res_overflows": 0,
        "data_num": 0,
    }
    
    # 根据数据总数和每个子进程处理的数据量，计算需要多少个子进程
    for i in range(math.ceil(all_num / each_num)):
        while True:
            # 统计当前正在运行的子进程数
            run_num = sum(1 for job in jobs if job.poll() is None)
            # 如果当前运行的子进程数小于最大允许的进程数，则跳出循环
            if run_num < max_num:
                break
            # 如果当前运行的子进程数达到最大值，则等待1秒后再检查
            time.sleep(1)
        
        # 计算当前子进程处理的数据范围
        start = i * each_num
        end = min((i + 1) * each_num, all_num)
        
        # 启动一个新的子进程来处理数据，捕获 stdout 和 stderr
        p = subprocess.Popen(
            ["python", file_name, str(start), str(end), 'false'],       # 最后一个参数是告诉 dataset_parallel.py 不要保存结果，只打印结果
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # 确保输出为字符串而不是字节
        )
        
        # 将子进程对象添加到列表中
        jobs.append(p)
        print(f"启动子进程处理数据范围: {start} - {end}")
        
        # 等待1秒后再启动下一个子进程
        time.sleep(1)
    
    # 等待所有子进程完成并收集输出
    for p in jobs:
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            print(f"子进程出错，错误信息: {stderr}", file=sys.stderr)
            continue
        
        try:
            result = json.loads(stdout)
            aggregated_result = aggregate_results(aggregated_result, result)
        except json.JSONDecodeError:
            print(f"无法解析子进程输出: {stdout}", file=sys.stderr)
    
    # 输出最终的聚合结果
    print("聚合结果:", json.dumps(aggregated_result, indent=2))

if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage: python count_overflow.py", file=sys.stderr)
        sys.exit(1)
    
    file_name = 'dataset_parallel.py'
    main(file_name)