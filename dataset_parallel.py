import sys
import torch
from collections import Counter
import torch.utils.data as data
import random
import pickle
import numpy as np
# import h5py
from tqdm import tqdm
import json
# from scipy import sparse
# from transformers import AutoTokenizer
# import javalang
from os import listdir
from os.path import isfile, join
import json
import subprocess
import re
import os
import copy
import math
from transformers import RobertaTokenizer, T5Model, T5ForConditionalGeneration, AdamW

# 模型类型设定为 CodeT5 的小模型
model_type = 'Salesforce/codet5-small'
local_path = './codet5/codet5-small'

# 初始化对应的分词器
# tokenizer = RobertaTokenizer.from_pretrained(model_type)
tokenizer = RobertaTokenizer.from_pretrained(local_path)

class dotdict(dict):
    """
    使用字典的方式访问属性的简单类
    可以通过属性访问（.）直接获取字典中的值
    """
    def __getattr__(self, name):  
        return self[name]

space_token = 'Ġ'

# 模拟的参数设置，使用dotdict进行管理
args = dotdict({
    'model_type': model_type,
    'max_conflict_length': 500,
    'max_resolve_length': 200,
})

# 自定义的特殊tokens，用于表示冲突时的括号分隔符
brackets_tokens = ['<lbra>', '<mbra>', '<rbra>']
# 将自定义tokens添加到tokenizer的词表中，succeed_num是成功添加的token数量
succeed_num = tokenizer.add_tokens(brackets_tokens)
assert succeed_num == len(brackets_tokens)

class Dataset(data.Dataset):
    """
    自定义数据集类，用于处理合并冲突文件数据并准备训练所需要的张量输入
    """
    def __init__(self, args, tokenizer, process_start, process_end):
        # 最大冲突长度与最大解决长度设定（来自args）
        self.max_conflict_length = args.max_conflict_length
        self.max_resolve_length = args.max_resolve_length
        self.tokenizer = tokenizer
        self.start = process_start
        self.end = process_end
        
        # 自定义冲突括号token
        self.lbra_token = '<lbra>'
        self.rbra_token = '<rbra>'

        # 如果不存在PROCESSED文件夹，则创建
        if not os.path.exists('PROCESSED'):
            os.mkdir('PROCESSED')
        
        # 处理后的数据保存路径
        self.data_path = 'PROCESSED/processed_%s_%s.pkl' % (self.start, self.end)

        # 原始数据所在路径
        total_raw_data_path = 'RAW_DATA'
        # 从JSON中加载原始数据（base、a、b分支代码及最终resolve结果）
        all_raw_base, all_raw_a, all_raw_b, all_raw_res = json.load(open('%s/raw_data' % (total_raw_data_path)))
        
        # 对原始数据进行处理
        self.process_data(all_raw_base, all_raw_a, all_raw_b, all_raw_res)

    def process_data(self, all_raw_base, all_raw_a, all_raw_b, all_raw_res):
        """
        将原始数据tokenize、对齐、padding并保存为pickle文件。
        """
        data_num = self.end - self.start
        max_conflict_length = 0
        max_resolve_length = 0
        inputs = []
        outputs = []

        # 遍历指定数据片段范围内的数据
        for i in tqdm(range(self.start, self.end)):

            self.ii = i
            raw_base = all_raw_base[i]
            raw_a = all_raw_a[i]
            raw_b = all_raw_b[i]
            raw_res = all_raw_res[i]

            # 对原始字符串进行简单清洗（多余空格）
            raw_base = ' '.join(raw_base.split())
            raw_a = ' '.join(raw_a.split())
            raw_b = ' '.join(raw_b.split())
            raw_res = ' '.join(raw_res.split())

            # 利用分词器对各版本代码进行分词
            tokens_base = self.tokenizer.tokenize(raw_base)
            tokens_a = self.tokenizer.tokenize(raw_a)
            tokens_b = self.tokenizer.tokenize(raw_b)
            tokens_res = self.tokenizer.tokenize(raw_res)

            # 将三个版本（base、a、b）的token通过git_merge方法合并得到合并冲突表示的token序列
            tokens_input = self.git_merge(tokens_base, tokens_a, tokens_b)

            # 将tokens转为对应的ids
            ids_input = self.tokenizer.convert_tokens_to_ids(tokens_input)
            ids_res = self.tokenizer.convert_tokens_to_ids(tokens_res)

            # 构建输入与输出序列（输出在前后添加bos和eos）
            cur_input = ids_input
            cur_output = [self.tokenizer.bos_token_id] + ids_res + [self.tokenizer.eos_token_id]

            # 更新当前数据集中最大长度统计
            max_conflict_length = max(max_conflict_length, len(cur_input))
            max_resolve_length = max(max_resolve_length, len(cur_output))

            # 对输入与输出进行padding到设定的最大长度
            cur_input = self.pad_length(cur_input, self.max_conflict_length, self.tokenizer.pad_token_id)
            cur_output = self.pad_length(cur_output, self.max_resolve_length, self.tokenizer.pad_token_id)
            inputs.append(cur_input)
            outputs.append(cur_output)
  
        # 打印最大序列长度信息
        print('max_conflict_length, max_resolve_length', max_conflict_length, max_resolve_length)
        print('all data num:%d remaining num:%d' % (data_num, len(inputs)))
        # 确保处理数据数量与期望一致
        assert data_num == len(inputs)

        data_num = len(inputs)
        # 将处理好的数据打包
        batches = [np.array(inputs), np.array(outputs)]

        # 将处理后的数据以pickle格式存储
        pickle.dump(batches, open(self.data_path, 'wb'))

    def pad_length(self, tokens, max_length, pad_id):
        """
        对token序列进行padding或截断，使其长度变为max_length。
        """
        if len(tokens) <= max_length:
            tokens = tokens + [pad_id] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        assert len(tokens) == max_length
        return tokens

    def __getitem__(self, offset):
        """
        根据index(offset)返回对应的数据item。
        """
        data = []
        # ? self.data 是哪里赋值的？    这个类没有被用于训练，只是用于收集数据
        for i in range(len(self.data)):
            if type(self.data[i][offset]) == np.ndarray:
                # 若为numpy数组直接加入列表
                data.append(self.data[i][offset])
            else:
                # 若为稀疏矩阵，转换为array后加入列表
                data.append(self.data[i][offset].toarray())  
        return data

    def __len__(self):
        """
        返回数据集中样本数量。
        """
        return len(self.data[0])

    def git_merge(self, tokens_base, tokens_a, tokens_b):
        """
        使用git merge-file命令对base、a、b三个版本进行三方合并，
        解析git产生的冲突标记并将结果转化为对应的token序列格式。
        """
        # 为当前索引的数据创建一个对应的文件夹，用于存放临时文件
        merge_path = 'GIT_MERGE_FILES/%s' % self.ii
        if not os.path.exists(merge_path):
            os.makedirs(merge_path)

        # 将tokens写入临时文件中，以便git合并工具使用
        with open('%s/base' % merge_path, 'w') as f:
            f.write('\n'.join(tokens_base))
        with open('%s/a' % merge_path, 'w') as f:
            f.write('\n'.join(tokens_a))
        with open('%s/b' % merge_path, 'w') as f:
            f.write('\n'.join(tokens_b))
        
        # 调用git merge-file命令进行三方合并，并使用--diff3 -p输出合并结果
        self.execute_command('git merge-file -L a -L base -L b %s/a %s/base %s/b --diff3 -p > %s/merge' 
                             % (merge_path, merge_path, merge_path, merge_path))
        
        # 读取合并结果文件
        merge_res = open('%s/merge' % merge_path).read().splitlines()
        merge_res = [x.strip() for x in merge_res if x.strip()]

        # 合并文件中会包含冲突标记行，如 "<<<<<<< a", ">>>>>>> b", "||||||| base", "======="
        # 找到这些标记行所在的索引，以用于解析冲突段落
        format_ids = [k for k, x in enumerate(merge_res) if x == '<<<<<<< a' or x == '>>>>>>> b' or x == '||||||| base' or x == '=======']
        assert len(format_ids) % 4 == 0

        final_tokens = []
        start = 0
        # 每4个特殊标记为一组： '<<<<<<< a', '||||||| base', '=======', '>>>>>>> b'
        # 用于确定一个冲突段落
        for k, x in enumerate(format_ids):
            if k % 4 == 0:
                assert (merge_res[format_ids[k]] == '<<<<<<< a' and 
                        merge_res[format_ids[k + 1]] == '||||||| base' and 
                        merge_res[format_ids[k + 2]] == '=======' and 
                        merge_res[format_ids[k + 3]] == '>>>>>>> b')
                
                # 上下文部分（在'<<<<<<< a'之前的非冲突内容）
                context_tokens = merge_res[start:format_ids[k]]
                # 来自a版本的代码片段
                a_tokens = merge_res[format_ids[k] + 1:format_ids[k + 1]]
                # base版本的代码片段
                base_tokens = merge_res[format_ids[k + 1] + 1:format_ids[k + 2]]
                # 来自b版本的代码片段
                b_tokens = merge_res[format_ids[k + 2] + 1:format_ids[k + 3]]

                start = format_ids[k + 3] + 1

                # 将冲突的三个片段用自定义的括号token以及sep_token分隔开
                # 格式：上下文 + [<lbra>] + a代码 + [sep] + base代码 + [sep] + b代码 + [<rbra>]
                final_tokens += context_tokens + [self.lbra_token] + a_tokens + [self.tokenizer.sep_token] + base_tokens + [self.tokenizer.sep_token] + b_tokens + [self.rbra_token]

        # 若还有未处理的尾部内容追加到final_tokens
        if start != len(merge_res):
            final_tokens += merge_res[start:]

        # 在最终序列的首尾加入bos和eos
        final_tokens = [self.tokenizer.bos_token] + final_tokens + [self.tokenizer.eos_token]
        
        return final_tokens

    def execute_command(self, cmd):
        """
        执行shell命令的辅助函数
        """
        p = subprocess.Popen(cmd, shell=True)
        p.wait()


if __name__ == '__main__':
    # 主执行入口，根据命令行参数获取处理数据的起始与结束索引
    process_start = int(sys.argv[1])
    process_end = int(sys.argv[2])
    # 实例化Dataset类进行数据处理
    dataset = Dataset(args, tokenizer, process_start, process_end)