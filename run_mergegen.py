# from cmath import inf
import os
from transformers import RobertaTokenizer, T5ForConditionalGeneration, AdamW
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from torch import optim
from torch.nn import functional as F
import torch
import torch.nn as nn
import json
import numpy as np
import random
from dataset import Dataset
import sys 
from torch.utils.data import DataLoader
import math
from accelerate import Accelerator
import time
from tqdm import tqdm
from accelerate import DistributedDataParallelKwargs

# 定义空格标识符，与tokenizer相关的特殊token
space_token = 'Ġ'



# 模型类型为Salesforce的CodeT5-small
# model_type = 'Salesforce/codet5-small'
model_type = './codet5/codet5-small'
# 初始化Roberta分词器，适用于Codet5
tokenizer = RobertaTokenizer.from_pretrained(model_type)

# 添加自定义的括号tokens，标识冲突中的A、B、base间的分隔
brackets_tokens = ['<lbra>', '<mbra>', '<rbra>']
succeed_num = tokenizer.add_tokens(brackets_tokens)
assert succeed_num == len(brackets_tokens)

# 初始化加速器，支持分布式训练和混合精度
accelerator = Accelerator()
# beam搜索的宽度，用于测试阶段生成序列
beam_num = 3

# 检测GPU可用性并获取可用GPU数量
use_cuda = torch.cuda.is_available()
device_ids = list(range(torch.cuda.device_count()))

class dotdict(dict):
    """
    dotdict允许通过属性访问字典内容，如x.key而不是x['key']
    """
    def __getattr__(self, name):  
        return self[name]

# 配置训练和测试参数
args = dotdict({
    'batch_size':35,
    'test_batch_size':30,
    'epoches':100,
    'lr':1e-4,
    'model_type':model_type,
    'max_conflict_length':500,
    'max_resolve_length':200,
})

def seed_everything(seed=0):
    """
    固定随机种子，确保实验的可重复性
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def save_model(model,save_path):
    """
    保存模型权重至指定路径
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path)

def get_tensor(data):
    """
    将numpy数据转换为torch张量
    """
    tensor = data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    return tensor

class MergeT5(nn.Module):
    """
    定义用于合并冲突的T5模型类，对输入（合并冲突的代码）和输出（解决后的代码）进行建模
    """
    def __init__(self, args):
        super(MergeT5, self).__init__()
        # 使用预训练的T5ForConditionalGeneration模型
        self.t5 = T5ForConditionalGeneration.from_pretrained(args.model_type)
        # 调整词表大小，包含新增的自定义tokens
        self.t5.resize_token_embeddings(len(tokenizer))
        self.embedding_dim = self.t5.config.hidden_size

    def forward(self, input_txt, output_txt, stage):
        """
        （teacher forcing）前向计算过程：
        input_txt: 输入序列（合并冲突后的代码片段）
        output_txt: 目标输出序列（解决冲突后的正确代码）
        stage: 当前阶段（train/dev），决定返回值。测试阶段在另一个函数中实现。
        """
        # 构建attention mask，pad_token_id的位置不需要关注
        attention_mask = input_txt != tokenizer.pad_token_id
        
        # 使用T5编码器对输入进行编码
        input_em = self.t5.encoder(input_ids=input_txt, 
                                   attention_mask=attention_mask, 
                                   return_dict=True)['last_hidden_state']
        # 使用T5解码器和编码结果生成输出logits
        logits = self.t5.decoder(input_ids=output_txt, 
                                 encoder_hidden_states=input_em, 
                                 encoder_attention_mask=attention_mask, 
                                 return_dict=True)['last_hidden_state']
        
        # 对logits进行缩放，然后通过lm_head映射到词表维度
        logits = logits * (self.embedding_dim ** -0.5)
        logits = self.t5.lm_head(logits)
        
        # 使用softmax转成概率分布，再取log（log_softmax）
        outputs = F.softmax(logits, dim=-1) 
        outputs = torch.log(outputs.clamp(min=1e-10, max=1))            # clamp 用于确保在计算对数概率时，输入值不会低于 1e-10 或超过 1，以避免数值不稳定性（如 log(0) 导致的负无穷）：

        # 构建label，将output_txt右移一位作为预测目标
        # pad_token_id用于对齐，label中0位置需要注意进行mask
        label = output_txt
        label = torch.cat([label, torch.ones(len(label), 1).cuda(input_txt.device) * tokenizer.pad_token_id], dim=-1)   # 在序列最后添加一个pad_token_id
        label = label[:,1:]         # 右移一位，去除第一个token，也就是<s>，因为这个token不需要预测
        label = label.long()
        mask = label != 0  # 非pad位置才计算loss

        # 使用NLL loss计算损失
        loss = F.nll_loss(outputs.view(-1, outputs.size(-1)), 
                          label.contiguous().view(-1), 
                          reduction='none')                     # 不对损失进行缩减，保留每个位置的损失值。
        loss = loss.masked_fill(mask.view(-1)==False, 0)

        if stage == 'train':
            # 返回损失总和和有效mask的数量（用于计算平均loss）
            return loss.sum(), mask.sum()
        elif stage == 'dev' or stage == 'test':
            # 返回预测结果、损失和mask数量、label用于评价
            return torch.argmax(outputs, dim=-1), loss.sum(), mask.sum(), label

def train(accelerator, model, train_loader, optimizer, epoch, best_acc, dev_loader, f):
    """
    训练过程:
    1. model.train() 
    2. 对每个batch计算loss并进行梯度回传与优化更新
    3. 每个epoch结束后在dev集上评估，更新best_acc
    """
    model.train()
    total_data = 0
    total_loss = 0

    for idx, batch in enumerate(tqdm(train_loader)):
        
        assert isinstance(batch, list)
        for i in range(len(batch)):
            batch[i] = get_tensor(batch[i])

        # 计算loss
        loss, mask = model(batch[0], batch[1], 'train')
        loss = loss.sum() / mask.sum()

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        total_data += len(batch[0])
        total_loss += loss.item()

    accelerator.print("epoch: %d batch: %d/%d  data: %d/%d loss: %.4f device:%s\n" 
                      % (epoch, idx, len(train_loader), total_data, len(train_loader.dataset), total_loss / len(train_loader), loss.device))

    flag_bigger = None
    if accelerator.is_main_process:
        # 在dev集上评估
        exactly_match_num, exactly_match_ids, total_dev_output = dev(model, dev_loader, epoch)
        exactly_match_ratio = exactly_match_num / len(dev_loader.dataset)
        f.write('epoch: {} exactly match:{} is better: {}\n'.format(epoch, exactly_match_ratio, exactly_match_ratio > best_acc))
        f.flush()

        if not os.path.exists('OUTPUT/DEV_OUTPUT'):
            os.makedirs('OUTPUT/DEV_OUTPUT')
        flag_bigger = exactly_match_ratio > best_acc
        # 如果比之前的best更好，则保存模型和输出
        if exactly_match_ratio > best_acc:
            best_acc = exactly_match_ratio
            torch.save(model.module.state_dict(),"best_model.pt")
            for k in range(len(total_dev_output)):
                p = open('OUTPUT/DEV_OUTPUT/%s'%(k),'w')
                p.write(total_dev_output[k])
                p.close()

    accelerator.wait_for_everyone()
    return best_acc, flag_bigger

def see_results(inputs, outputs, targets):
    """
    将模型输出ids转化为对应的字符串，并与参考结果对比
    用于检查模型的生成结果
    """
    assert len(outputs.shape) == 3

    inputs = inputs.cpu().numpy()
    if len(outputs.shape) == 2:
        outputs = outputs.unsqueeze(1)
    outputs = outputs.cpu().numpy()
    targets = targets.cpu().numpy()

    input_strings = []
    output_strings = []
    target_strings = []
    input_token_len = []
    ref_token_len = []
    for i in range(len(outputs)):
        beam_output_strings = []
        for j in range(len(outputs[i])):
            cur_output = outputs[i][j].tolist()
            if tokenizer.eos_token_id in cur_output:
                cur_output = cur_output[:cur_output.index(tokenizer.eos_token_id)] 
            output_token = tokenizer.convert_ids_to_tokens(cur_output)
            output_string = ''.join(output_token).replace(space_token, " ")
            beam_output_strings.append(output_string)

        cur_target = targets[i].tolist()[1:]
        cur_input = inputs[i].tolist()[1:]
        if tokenizer.pad_token_id in cur_input:
            cur_input = cur_input[:cur_input.index(tokenizer.pad_token_id)]
        input_token_len.append(len(cur_input))
        if tokenizer.eos_token_id in cur_target:
            cur_target = cur_target[:cur_target.index(tokenizer.eos_token_id)] 
        ref_token_len.append(len(cur_target))
        ref_token = tokenizer.convert_ids_to_tokens(cur_target)
        ref_string = ''.join(ref_token).replace(space_token, " ")
        input_token = tokenizer.convert_ids_to_tokens(cur_input)
        input_string = ''.join(input_token).replace(space_token, " ")

        output_strings.append(beam_output_strings)
        target_strings.append(ref_string)
        input_strings.append(input_string)
    return input_strings, output_strings, target_strings, input_token_len, ref_token_len

def dev(model, val_loader, epoch, dev_type='train'):
    """
    在验证集或开发集上进行模型评估。
    计算exactly match的数量，并可选择性输出预测结果到指定文件夹。
    """
    all_index = json.load(open('all_index'))
    valid_index = all_index['valid']

    model.eval()
    total_dev_output = []
    
    total_data = 0
    total_loss = 0
    total_mask = 0
    exactly_match_num = 0
    exactly_match_ids = []
    total_results = None
    total_label = None

    total_input_strings = []
    total_output_strings = []
    total_target_strings = []
    total_output_tokens = []
    total_target_tokens = []

    for idx, batch in enumerate(tqdm(val_loader)):
        for i in range(len(batch)):
            batch[i] = get_tensor(batch[i])
            batch[i] = batch[i].cuda()
        
        with torch.no_grad():   
            # 调用模型forward
            output, loss, mask, label = model(batch[0], batch[1], 'dev')
            results = output == label
            if idx == 0:
                total_results = results.cpu()
                total_label = label.cpu()
            else:
                total_results = torch.cat((total_results, results.cpu()), dim=0)
                total_label = torch.cat((total_label, label.cpu()), dim=0)
            total_loss += loss.sum().item()
            total_mask += mask.sum().item()
            
            if dev_type == 'dev':
                # 将预测结果和参考结果转换为可读文本
                input_strings, output_strings, ref_strings, input_token_len, ref_token_len = see_results(batch[0], output, batch[1])
                total_output_strings.extend(output_strings)
                total_target_strings.extend(ref_strings)

        total_data += len(output)

    # 对所有结果判断每条数据是否完全match（全词准确）
    total_results = torch.masked_fill(total_results, total_label == 0, True)
    total_results = torch.all(total_results, dim=-1)
    exactly_match_ids = total_results.nonzero(as_tuple=False).tolist()
    exactly_match_num = total_results.sum().item()

    assert total_data == len(val_loader.dataset)

    if dev_type == 'dev':
        # 将成功与失败case分别保存
        output_path = 'OUTPUT/DEV_OUTPUT'
        if not os.path.exists('%s/SUCCEED'%output_path):
            os.makedirs('%s/SUCCEED'%output_path)
            os.makedirs('%s/FAIL'%output_path)
        for i in range(len(total_results)):
            if total_results[i]:
                p = open('%s/SUCCEED/%s'%(output_path, valid_index[i]), 'w')
            else:
                p = open('%s/FAIL/%s'%(output_path, valid_index[i]), 'w')

            p.write('---------------------------------\n')
            p.write('ref_dataloader:\n%s\n'%total_target_strings[i])
            p.write('---------------------------------\n')
            p.write('output:\n%s\n'%total_output_strings[i])
            p.write('---------------------------------\n')
            p.close()
        total_dev_output = total_output_strings

    return exactly_match_num, exactly_match_ids, total_dev_output

def test_beam(model, test_loader):
    """
    使用beam搜索在测试集上生成预测结果，并比较与参考的exact match情况。
    """
    all_index = json.load(open('all_index'))
    test_index = all_index['test']
    model.eval()
    total_test_output = []
    
    total_data = 0
    exactly_match_num = 0
    exactly_match_ids = []
    total_results = None
    total_label = None

    total_input_strings = []
    total_output_strings = []
    total_target_strings = []
    total_output_tokens = []
    total_target_tokens = []
    total_input_token_len = []
    total_ref_token_len = []

    for idx, batch in enumerate(tqdm(test_loader)):
        for i in range(len(batch)):
            batch[i] = get_tensor(batch[i])
            batch[i] = batch[i].cuda()
        with torch.no_grad():
            # 准备encoder输入和mask
            input_attention_mask = batch[0] != tokenizer.pad_token_id
            input_em = model.t5.encoder(input_ids=batch[0], attention_mask=input_attention_mask, return_dict=True)['last_hidden_state']
            input_em = BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=input_em)
            # 使用beam search进行生成，多序列输出
            beam_output = model.t5.generate(encoder_outputs=input_em, 
                                            attention_mask=input_attention_mask,
                                            decoder_input_ids=(torch.ones(len(batch[0]), 1) * tokenizer.bos_token_id).long().cuda(),
                                            num_beams=beam_num, 
                                            num_return_sequences=beam_num, 
                                            max_new_tokens=args.max_resolve_length * 2)
            beam_output = beam_output[:,1:]

            beam_output = beam_output.view(len(batch[0]), beam_num, -1)
            output = beam_output[:,0,:]

            # 截断到指定长度，并padding到max_resolve_length
            output = output[:,:args.max_resolve_length]
            output = torch.cat([output, torch.zeros(len(output), args.max_resolve_length - len(output[0])).long().cuda()], dim=-1)

            label = batch[1]
            # ? 这里生成的 output 第一个 token 是 <s>，所以 label 不需要右移一位，这样做对吗？？？
            # label = torch.cat([label, torch.zeros(len(label), 1).long().cuda(batch[0].device)], dim=-1)
            # label = label[:,1:]

            assert output.shape == label.shape

            results = output == label
            if idx == 0:
                total_results = results.cpu()
                total_label = label.cpu()
            else:
                total_results = torch.cat((total_results, results.cpu()), dim=0)
                total_label = torch.cat((total_label, label.cpu()), dim=0)

            input_strings, output_strings, ref_strings, input_token_len, ref_token_len = see_results(batch[0], beam_output, batch[1])
            total_output_strings.extend(output_strings)
            total_target_strings.extend(ref_strings)
            total_input_strings.extend(input_strings)
            total_input_token_len.extend(input_token_len)
            total_ref_token_len.extend(ref_token_len)
        total_data += len(output)

    total_results = torch.masked_fill(total_results, total_label == 0, True)
    total_results = torch.all(total_results, dim=-1)
    exactly_match_ids = total_results.nonzero(as_tuple=False).tolist()
    exactly_match_num = total_results.sum().item()

    assert total_data == len(test_loader.dataset)

    output_path = 'OUTPUT/test_withInputs_OUTPUT'
    if not os.path.exists('%s/SUCCEED'%output_path):
        os.makedirs('%s/SUCCEED'%output_path)
        os.makedirs('%s/FAIL'%output_path)
    for i in range(len(total_results)):
        if total_results[i]:
            p = open('%s/SUCCEED/%s'%(output_path, test_index[i]), 'w')
        else:
            p = open('%s/FAIL/%s'%(output_path, test_index[i]), 'w')

        p.write('input token length: %s\n'%total_input_token_len[i])
        p.write('input:\n%s\n'%total_input_strings[i])
        p.write('---------------------------------\n')
        p.write('ref_token_length: %s\n'%total_ref_token_len[i])
        p.write('ref_dataloader:\n%s\n'%total_target_strings[i])
        p.write('---------------------------------\n')
        p.write('output:\n%s\n'%total_output_strings[i])
        p.write('---------------------------------\n')
        p.close()
    total_test_output = total_output_strings
    return exactly_match_num, exactly_match_ids, total_test_output

def main_train():
    """
    主训练函数:
    1. 读取和创建训练与验证数据集
    2. 定义模型、优化器与加速器包装
    3. 加载已有的best_model（若有），获取初始best_acc
    4. 循环多epoch训练，并在dev集上评估
    5. 根据dev表现动态调整学习率和提前停止条件
    """
    open('train_state', 'w').write(str(1))
    if accelerator.is_main_process:
        train_set = Dataset(args, tokenizer, 'train')
        dev_set = Dataset(args, tokenizer, 'valid')
    accelerator.wait_for_everyone()
    
    # 确保每个进程都有数据集（这么写可能是出于 Dataset 构造函数可能会有下载内容？这样写只需要下载一遍？）
    train_set = Dataset(args, tokenizer, 'train')
    dev_set = Dataset(args, tokenizer, 'valid')
    start_time = time.time()
    
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dataset=dev_set, batch_size=args.test_batch_size)
    model = MergeT5(args)
    optimizer = AdamW(model.parameters(), args.lr)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # 若已存在best_model.pt，则加载
    if os.path.exists("best_model.pt"):
        model.module.load_state_dict(torch.load("best_model.pt", map_location=torch.device('cpu')))
        print('model loaded!')
    if accelerator.is_main_process:
        if os.path.exists("best_model.pt"):
            exactly_match_num, _, _ = dev(model, dev_loader, -1)
            best_acc = exactly_match_num / len(dev_loader.dataset)
        else:
            best_acc = -1
        open('best_acc', 'w').write(str(best_acc))
    accelerator.wait_for_everyone()
    best_acc = float(open('best_acc', 'r').read())
    print(best_acc)

    f = open('OUTPUT/train_process','a')
    
    small_num = 0
    try_num = 0
    max_try_num = 3
    max_small_num = 6
    for epoch in range(args.epoches):
        # 若train_state设为0则停止训练
        if int(open('train_state').read()) == 0:
            break
        best_acc, flag_bigger = train(accelerator, model, train_loader, optimizer, epoch, best_acc, dev_loader, f)
        if accelerator.is_main_process:
            # 如果新的dev表现更好，small_num清0；否则计数+1
            if flag_bigger == True:
                small_num = 0
            else:
                small_num += 1
                # 若连续max_small_num个epoch没提升，则减半LR并从best恢复模型
                if small_num == max_small_num:
                    small_num = 0
                    try_num += 1
                    model.module.load_state_dict(torch.load("best_model.pt", map_location=torch.device('cpu')))
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr'] * 0.5
                # 若已尝试max_try_num次还没提升则停止训练
                if try_num == max_try_num:
                    open('train_state', 'w').write(str(0))
        accelerator.wait_for_everyone()
    accelerator.wait_for_everyone()
    end_time = time.time()
    if accelerator.is_main_process:
        f.write("time: %sh"%((end_time - start_time) / 3600))
        f.close()

def main_dev():
    """
    主验证函数:
    使用best_model.pt在验证集上评估并存储生成结果
    """
    dev_set = Dataset(args, tokenizer, 'valid')
    dev_loader = DataLoader(dataset=dev_set, batch_size=args.batch_size)
    model = MergeT5(args)
    model.load_state_dict(torch.load("best_model.pt"))

    # 似乎应该注释掉，防止和 Accelerator 重复
    if use_cuda:
        model = nn.DataParallel(model, device_ids = device_ids)
        model = model.cuda(device_ids[0])
    best_acc = -1
    exactly_match_num, exactly_match_ids, total_dev_output = dev(model, dev_loader, -1, dev_type='dev')

    json.dump(total_dev_output, open('OUTPUT/total_gen_output_dev.json', 'w'))

    with open('OUTPUT/dev_process', 'w') as f:
        f.write('exactly match: %f\n'%(exactly_match_num / len(dev_loader.dataset)))

def main_test():
    """
    主测试函数:
    使用best_model.pt在测试集上评估表现，并使用beam搜索生成预测结果。
    """
    dev_set = Dataset(args, tokenizer, 'valid')
    dev_loader = DataLoader(dataset=dev_set, batch_size=args.batch_size)

    test_set = Dataset(args, tokenizer, 'test')
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size)
    model = MergeT5(args)
    model.load_state_dict(torch.load("best_model.pt"))
    if use_cuda:
        model = model.cuda(device_ids[0])
    f =  open('OUTPUT/test_process', 'w')
    dev_exactly_match_num, _, _ = dev(model, dev_loader, -1)
    f.write('dev_exactly_match: %f\n'%(dev_exactly_match_num / len(dev_loader.dataset)))
    f.flush()
    exactly_match_num, exactly_match_ids, total_test_output = test_beam(model, test_loader)

    json.dump(total_test_output, open('OUTPUT/total_gen_output_test_beam.json', 'w'))
    # make sure RESULTS folder exists
    if not os.path.exists('RESULTS'):
        os.makedirs('RESULTS')
    json.dump(exactly_match_ids, open('RESULTS/test_gen_exactly_match_ids', 'w'))

    f.write('test_exactly_match %s out of %s\n'%(exactly_match_num, len(test_loader.dataset)))
    f.write('test exactly match: %f\n'%(exactly_match_num / len(test_loader.dataset)))
    f.close()

def debug():
    dev_set = Dataset(args, tokenizer, 'valid')
    dev_loader = DataLoader(dataset=dev_set, batch_size=1)
    for idx, batch in enumerate(dev_loader):
        print(tokenizer.decode(batch[0][0]))
        print('---------------------------------')
        print(tokenizer.decode(batch[1][0]))
        print()
        print()

if __name__ == '__main__':
    # 从命令行参数中获取当前执行阶段（'train'、'dev'或'test'）
    stage = str(sys.argv[1])

    # 固定随机种子
    seed_everything(0)
    if not os.path.exists('OUTPUT'):
        os.makedirs('OUTPUT')
    # 根据stage选择执行哪一部分
    if stage == 'train':
        main_train()
    elif stage == 'dev':
        main_dev()
    elif stage== 'test':
        main_test()
    else:
        debug()