# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer

import torch
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser, AutoConfig, GenerationConfig
from data_utils import train_info_args, NN_DataHelper, get_deepspeed_config, build_messages
from aigc_zoo.model_zoo.xverse.llm_model import MyTransformer

deep_config = get_deepspeed_config()


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments,))
    (model_args,)  = parser.parse_dict(train_info_args, allow_extra_keys=True)

    dataHelper = NN_DataHelper(model_args)
    tokenizer, _, _,_= dataHelper.load_tokenizer_and_config()
    

    config = AutoConfig.from_pretrained('./best_ckpt')
    pl_model = MyTransformer(config=config, model_args=model_args,torch_dtype=config.torch_dtype,)

    # deepspeed 权重使用转换脚本命令
    # 一般根据时间排序选最新的权重文件夹
    # cd best_ckpt/last
    # python zero_to_fp32.py . ../last.ckpt

    train_weight = './best_ckpt/last.ckpt'
    pl_model.load_sft_weight(train_weight,strict=True)

    # 保存hf权重
    # config.save_pretrained('convert/')

    # 保存sft p-tuning-v2 权重
    #  pl_model.save_sft_weight('convert/pytorch_model_sft_ptv2.bin')

    # 保存sft权重
    # pl_model.save_sft_weight('convert/pytorch_model_sft.bin')

    model = pl_model.get_llm_model()

    model.eval().half().cuda()

    text_list = ["写一个诗歌，关于冬天",
                 "晚上睡不着应该怎么办",
                 "从南京到上海的路线",
                 ]

    generation_config = GenerationConfig(**{
        "pad_token_id": 1,
        "bos_token_id": 2,
        "eos_token_id": 3,
        "max_new_tokens": 512,
        "temperature": 0.5,
        "top_k": 30,
        "top_p": 0.85,
        "repetition_penalty": 1.1,
        "do_sample": True,
    })

    for input in text_list:
        messages = build_messages(input)
        response = model.chat(tokenizer=tokenizer,messages=messages,generation_config=generation_config)
        print('input', input)
        print('output', response)