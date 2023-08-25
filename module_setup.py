# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/8 9:55
from deep_training.utils.hf import register_transformer_model,register_transformer_config
# from deep_training.nlp.models.xverse.modeling_xverse import XverseForCausalLM,XverseConfig
from aigc_zoo.model_zoo.xverse.llm_model import MyXverseForCausalLM,XverseConfig
from transformers import AutoModelForCausalLM

__all__ = [
    "module_setup"
]




def module_setup():
    # 导入模型
    register_transformer_config(XverseConfig)
    register_transformer_model(MyXverseForCausalLM, AutoModelForCausalLM)