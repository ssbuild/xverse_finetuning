# -*- coding: utf-8 -*-
# @Time:  23:20
# @Author: tk
# @File：model_maps
from aigc_zoo.constants.define import (TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING)

__all__ = [
    "TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING",
    "train_model_config"
]

train_info_models = {
    'XVERSE-13B-Chat': {
        'model_type': 'xverse',
        'model_name_or_path': '/data/nlp/pre_models/torch/xverse/XVERSE-13B-Chat',
        'config_name': '/data/nlp/pre_models/torch/xverse/XVERSE-13B-Chat/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/xverse/XVERSE-13B-Chat',
    },

    'xverse-13b-chat-int4': {
        'model_type': 'xverse',
        'model_name_or_path': '/data/nlp/pre_models/torch/xverse/xverse-13b-chat-int4',
        'config_name': '/data/nlp/pre_models/torch/xverse/xverse-13b-chat-int4/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/xverse/xverse-13b-chat-int4',
    },

    'XVERSE-13B': {
        'model_type': 'xverse',
        'model_name_or_path': '/data/nlp/pre_models/torch/xverse/XVERSE-13B',
        'config_name': '/data/nlp/pre_models/torch/xverse/XVERSE-13B/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/xverse/XVERSE-13B',
    },

    'xverse-13b-int4': {
        'model_type': 'xverse',
        'model_name_or_path': '/data/nlp/pre_models/torch/xverse/xverse-13b-int4',
        'config_name': '/data/nlp/pre_models/torch/xverse/xverse-13b-int4/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/xverse/xverse-13b-int4',
    },


}


# 按需修改
# TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING

train_model_config = train_info_models['XVERSE-13B-Chat']