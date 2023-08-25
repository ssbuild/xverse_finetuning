# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer
import torch
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser, GenerationConfig
from data_utils import train_info_args, NN_DataHelper, get_deepspeed_config,build_messages
from aigc_zoo.model_zoo.xverse.llm_model import MyTransformer


deep_config = get_deepspeed_config()

if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments,))
    (model_args,)  = parser.parse_dict(train_info_args, allow_extra_keys=True)

    dataHelper = NN_DataHelper(model_args)
    tokenizer, config, _,_= dataHelper.load_tokenizer_and_config()

    pl_model = MyTransformer(config=config, model_args=model_args,torch_dtype=config.torch_dtype,)
    model = pl_model.get_llm_model()
    model = model.eval()
    if hasattr(model,'quantize'):
        # 量化
        if not model.quantized:
            # 按需修改，目前只支持 4/8 bit 量化 ， 可以保存量化模型
            model.half().quantize(4).cuda()
            # 保存量化权重
            # model.save_pretrained('xverse-13b-chat-int4',max_shard_size="2GB")
            # exit(0)
        else:
            # 已经量化
            model.half().cuda()
    else:
        model.half().cuda()

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
        response = model.chat(tokenizer=tokenizer,messages=messages, generation_config=generation_config)
        print('input', input)
        print('output', response)