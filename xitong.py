import os
import torch
from transformers import AutoModel, AutoTokenizer
import streamlit as st
from PIL import Image
from dataclasses import asdict

@st.cache_resource
def load_multimodel():
# 初始化模型和分词器
    torch.set_grad_enabled(False)
    model_path_multi = "/mnt/network_share/任意/医疗大模型/完整医疗大模型系统/多模态大模型/yiliaoduomotai01"
    devices = torch.device('cuda:1')
    multimodel = AutoModel.from_pretrained(
        model_path_multi,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(devices).eval()

    multitokenizer = AutoTokenizer.from_pretrained(model_path_multi, trust_remote_code=True)
    return multimodel,multitokenizer

# 定义存储图片的文件夹
image_dir = "images"
os.makedirs(image_dir, exist_ok=True)

# 定义聊天逻辑
def chat_with_model(image,multimodel,multitokenizer):
    history = []
    if image is not None:
        image_path = os.path.join(image_dir, "uploaded_image.png")
        image.save(image_path)
        text = "请根据以上图片给出详细医疗报告"
        # 模型解析图像信息
        text = "<ImageHere> " + text
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                image_response, _ = multimodel.chat(multitokenizer, query=text, image=image_path, history=history, do_sample=False)
        text = f"用户的医疗图片信息：{image_response}\n"
        return text
    else:
        return "用户提问：没有上传图片"


# isort: skip_file
import numpy as np
import copy
import json
import pickle
import re
import warnings
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification
import streamlit as st
import torch
from torch import nn
from transformers.generation.utils import (LogitsProcessorList,
                                       StoppingCriteriaList)
import time
from transformers.utils import logging
from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM  # isort: skip
from peft import AutoPeftModelForCausalLM
from peft import PeftModel
logger = logging.get_logger(__name__)
device = torch.device("cuda:0")


@dataclass
class GenerationConfig:
    # this config is used for chat to provide more diversity
    max_length: int = 32768
    top_p: float = 0.8
    temperature: float = 0.8
    do_sample: bool = True
    repetition_penalty: float = 1.005


@torch.inference_mode()
def generate_interactive(
        model,
        tokenizer,
        prompt,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor],
                                                    List[int]]] = None,
        additional_eos_token_id: Optional[int] = None,
        **kwargs,
):
    inputs = tokenizer([prompt], padding=True, return_tensors='pt')
    input_length = len(inputs['input_ids'][0])
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    input_ids = inputs['input_ids']
    _, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]
    if generation_config is None:
        generation_config = model.generation_config
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)
    bos_token_id, eos_token_id = (  # noqa: F841  # pylint: disable=W0612
        generation_config.bos_token_id,
        generation_config.eos_token_id,
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    if additional_eos_token_id is not None:
        eos_token_id.append(additional_eos_token_id)
    has_default_max_length = kwargs.get(
        'max_length') is None and generation_config.max_length is not None
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using 'max_length''s default ({repr(generation_config.max_length)}) \
                to control the generation length. "
            'This behaviour is deprecated and will be removed from the \
                config in v5 of Transformers -- we'
            ' recommend using `max_new_tokens` to control the maximum \
                length of the generation.',
            UserWarning,
        )
    elif generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + \
                                       input_ids_seq_length
        if not has_default_max_length:
            logger.warn(  # pylint: disable=W4902
                f"Both 'max_new_tokens' (={generation_config.max_new_tokens}) "
                f"and 'max_length'(={generation_config.max_length}) seem to "
                "have been set. 'max_new_tokens' will take precedence. "
                'Please refer to the documentation for more information. '
                '(https://huggingface.co/docs/transformers/main/'
                'en/main_classes/text_generation)',
                UserWarning,
            )

    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = 'input_ids'
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, "
            f"but 'max_length' is set to {generation_config.max_length}. "
            'This can lead to unexpected behavior. You should consider'
            " increasing 'max_new_tokens'.")

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None \
        else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None \
        else StoppingCriteriaList()

    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config,
        stopping_criteria=stopping_criteria)
    logits_warper = model._get_logits_warper(generation_config)

    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    scores = None
    while True:
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, **model_kwargs)
        # forward pass to get next token
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        if generation_config.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=False)
        unfinished_sequences = unfinished_sequences.mul(
            (min(next_tokens != i for i in eos_token_id)).long())

        output_token_ids = input_ids[0].cpu().tolist()
        output_token_ids = output_token_ids[input_length:]
        for each_eos_token_id in eos_token_id:
            if output_token_ids[-1] == each_eos_token_id:
                output_token_ids = output_token_ids[:-1]
        response = tokenizer.decode(output_token_ids)

        yield response
        # stop when each sentence is finished
        # or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(
                input_ids, scores):
            break


def on_btn_click():
    del st.session_state.messages


#
@st.cache_resource
def load_model():
    recall_model_path = "/mnt/network_share/任意/医疗大模型/完整医疗大模型系统/医疗预训练大模型/rag/train_similar_model/mixed_model_1"
    recall_tokenizer = AutoTokenizer.from_pretrained(recall_model_path, model_max_length=512)
    recall_model = AutoModel.from_pretrained(recall_model_path)
    rank_model_path = '/mnt/network_share/任意/医疗大模型/完整医疗大模型系统/医疗预训练大模型/rag/train_similar_model/best_model/rank'
    rank_model = BertForSequenceClassification.from_pretrained(
        rank_model_path)
    rank_tokenizer = AutoTokenizer.from_pretrained(rank_model_path, model_max_length=512)
    device = torch.device("cuda:0")
    rank_model = rank_model.to(device)
    recall_model = recall_model.to(device)

    llm_model_path = "/mnt/network_share/任意/model/internlm2-chat-7b"
    llm_model = AutoModel.from_pretrained(llm_model_path,
                                                     trust_remote_code=True).to(
        torch.bfloat16).to(device)
    # llm_model = PeftModel.from_pretrained(llm_model,  "/mnt/network_share/任意/医疗大模型/完整医疗大模型系统/医疗预训练大模型/yiliao_sft_yuan")
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path,
                                                  trust_remote_code=True)

    llm_model = llm_model.eval()
    print("模型加载完毕")

    return llm_model, llm_tokenizer, recall_tokenizer, recall_model, rank_model, rank_tokenizer


def prepare_generation_config():
    with st.sidebar:
        max_length = st.slider('Max Length',
                               min_value=8,
                               max_value=32768,
                               value=32768)
        top_p = st.slider('Top P', 0.0, 1.0, 0.8, step=0.01)
        temperature = st.slider('Temperature', 0.0, 1.0, 0.7, step=0.01)
        st.button('Clear Chat History', on_click=on_btn_click)

    generation_config = GenerationConfig(max_length=max_length,
                                         top_p=top_p,
                                         temperature=temperature)

    return generation_config


user_prompt = '<|im_start|>user\n{user}<|im_end|>\n'
robot_prompt = '<|im_start|>assistant\n{robot}<|im_end|>\n'
cur_query_prompt = '<|im_start|>user\n{user}<|im_end|>\n\
    <|im_start|>assistant\n'


def combine_history(prompt):
    messages = st.session_state.messages
    meta_instruction = ('You are InternLM (书生·浦语), a helpful, honest, '
                        'and harmless AI assistant developed by Shanghai '
                        'AI Laboratory (上海人工智能实验室).')
    total_prompt = f"<s><|im_start|>system\n{meta_instruction}<|im_end|>\n"
    for message in messages:
        cur_content = message['content']
        if message['role'] == 'user':
            cur_prompt = user_prompt.format(user=cur_content)
        elif message['role'] == 'robot':
            cur_prompt = robot_prompt.format(robot=cur_content)
        else:
            raise RuntimeError
        total_prompt += cur_prompt
    total_prompt = total_prompt + cur_query_prompt.format(user=prompt)
    return total_prompt

def combine_history2(prompt):
    meta_instruction = ('你是一个医学诊疗系统，你会结合所给病例，回答用户问题。')
    total_prompt = f"<s><|im_start|>system\n{meta_instruction}<|im_end|>\n"

    total_prompt = total_prompt + cur_query_prompt.format(user=prompt)
    return total_prompt



    response_encoded = recall_tokenizer(response, return_tensors='pt', padding=True, truncation=True)
    response_embedding = recall_model(**response_encoded).last_hidden_state[:, 0, :]


import re

def extract_keshi(text, llm_model, llm_tokenizer, recall_tokenizer, recall_model, rank_model, rank_tokenizer):
    print("-----------------------------------------------------------")
    print("开始抽取keshi")
    
    # 生成 prompt
    prompt = f"文本：{text}\n你的任务是识别文本的推荐的科室，并将推荐科室与上述科室类型列表的实际名称对齐，请输出对齐后的科室名称：(仅输出科室名称)\n科室类型列表：{keshi_dist}\n"
    
    # 调用模型获取推荐科室
    response_keshi = chat2(prompt, llm_model, llm_tokenizer, recall_tokenizer, recall_model, rank_model, rank_tokenizer)
    
    # 使用正则表达式提取科室名称
    match = re.search(r'科室名称：(.+)', response_keshi)
    if match:
        response_keshi = match.group(1).strip()
    else:
        response_keshi = response_keshi.strip()
    
    print("模型抽取科室：" + response_keshi)

    # 如果 response_keshi 在 keshi_dist 中，直接返回
    if response_keshi in keshi_dist:
        #print(response_keshi)
        return response_keshi

    # 遍历科室列表里的每个值看是否在 response_keshi 中
    found_keshi = None
    for pipei_keshi in keshi_dist:
        if pipei_keshi in response_keshi:
            if found_keshi is not None:
                found_keshi = None
                break
            found_keshi = pipei_keshi

    if found_keshi is not None:
        print(found_keshi)
        return found_keshi

    # 否则进行相似度计算
    best_match, score = process.extractOne(response_keshi, keshi_dist, scorer=fuzz.ratio)
    print(f"最相似的科室：{best_match}, 相似度：{score}")

    return best_match









def chat2(prompt,llm_model, llm_tokenizer, recall_tokenizer, recall_model, rank_model, rank_tokenizer):
    response, history = llm_model.chat(llm_tokenizer,
                                       prompt,
                                       history=[],
                                       num_beams=5,  # 返回3个序列
                                       early_stopping=False,
                                       do_sample=True,
                                       temperature=1.2,
                                       top_p=0.7,  #
                                       top_k=50  #
                                       )
    print(response)
    return response


def get_similar_query(query,llm_model, llm_tokenizer, recall_tokenizer, recall_model, rank_model, rank_tokenizer,num=3):
    print("重写")
    results = []
    for _ in range(0, num):
        # 大模型进行改写

        #prompt = f"{query}\n问题:您是一个问题重写器，它可以将输入问题转换为一个更好的版本，以优化网络搜索。查看输入并尝试推理基础语义意图/意义。请严格限制输出不超过50字。"
        prompt = f"{query}\n问题:您是一个问题重写器，它可以将输入问题转换为一个更好的版本，以优化网络搜索。请严格限制输出不超过50字。"
        response = chat2(prompt,llm_model, llm_tokenizer, recall_tokenizer, recall_model, rank_model, rank_tokenizer)
        results.append(response)
    print("重写完毕")
    return results


def read_knowledge(path):
    with open(path, 'r', encoding='utf-8') as file:
        id_desc = json.load(file)
    return id_desc


def normal(vector):
    vector = vector.tolist()[0]
    ss = sum([s ** 2 for s in vector]) ** 0.5
    return [s / ss for s in vector]


def get_vector(sentence,llm_model, llm_tokenizer, recall_tokenizer, recall_model, rank_model, rank_tokenizer):
    device = torch.device("cuda:0")    
    encoded_input = recall_tokenizer([sentence], padding=True, truncation=True, return_tensors='pt', max_length=512)
    encoded_input = encoded_input.to(device)

    # Compute token embeddings
    with torch.no_grad():
        model_output = recall_model(**encoded_input)
    # Perform pooling. In this case, mean pooling.
    sentence_embeddings = normal(model_output[1])
    sentence_embeddings = np.array([sentence_embeddings])
    return sentence_embeddings


def get_candidate(input, faiss_index,llm_model, llm_tokenizer, recall_tokenizer, recall_model, rank_model, rank_tokenizer ,num=20):
    print("召回")
    vector = get_vector(input,llm_model, llm_tokenizer, recall_tokenizer, recall_model, rank_model, rank_tokenizer)
    D, I = faiss_index.search(vector, num)
    D = D[0]
    I = I[0]
    indexs = []
    for d, i in zip(D, I):
        indexs.append(i)
    print("召回完毕")
    return indexs


def rank_sentence(query, sentences,llm_model, llm_tokenizer, recall_tokenizer, recall_model, rank_model, rank_tokenizer):
    X = [[query[0:200], sentence[0:200]] for sentence in sentences]
    X = rank_tokenizer(X, padding=True, truncation=True, max_length=512, return_tensors='pt')
    X = X.to(device)
    scores = rank_model(**X).logits
    scores = torch.softmax(scores, dim=-1).tolist()
    scores = [round(s[1], 3) for s in scores]
    return scores


def rag_recall(query, faiss_index, id_knowledge,llm_model, llm_tokenizer, recall_tokenizer, recall_model, rank_model, rank_tokenizer):
    similar_querys = get_similar_query(query,llm_model, llm_tokenizer, recall_tokenizer, recall_model, rank_model, rank_tokenizer)
    index_score = {}
    for input in [query] + similar_querys:
        indexs = get_candidate(input, faiss_index,llm_model, llm_tokenizer, recall_tokenizer, recall_model, rank_model, rank_tokenizer, num=30)
        sentences = [id_knowledge[str(index)] for index in indexs]
        scores = rank_sentence(input, sentences,llm_model, llm_tokenizer, recall_tokenizer, recall_model, rank_model, rank_tokenizer)
        for index, score in zip(indexs, scores):
            if score < 0.8:
                continue
            index_score[index] = index_score.get(str(index), 0.0) + score

    results = sorted(index_score.items(), key=lambda s: s[1], reverse=True)
    return results[0:3]


def get_prompt(recall_result, fangan_knowledge):
    prompt = ""
    # 知识的id，召回的分数
    for i, [recall_id, recall_score] in enumerate(recall_result):
        prompt += f"案例:{fangan_knowledge[recall_id]}\n"
        print(prompt)
        print("------------------------------------------")
    return prompt


def chongxie_zhaohui_jingpai(xuanzekeshi, question,llm_model, llm_tokenizer, recall_tokenizer, recall_model, rank_model, rank_tokenizer):
    with open(
            f"/mnt/network_share/任意/医疗大模型/完整医疗大模型系统/医疗预训练大模型/rag/对话数据大全/{xuanzekeshi}/id_vector",
            "rb") as f:
        faiss_index = pickle.load(f)
    id_knowledge = read_knowledge(
        f"/mnt/network_share/任意/医疗大模型/完整医疗大模型系统/医疗预训练大模型/rag/对话数据大全/{xuanzekeshi}/id_desc_map.json")
    fangan_knowledge = []
    with open(
            f"/mnt/network_share/任意/医疗大模型/完整医疗大模型系统/医疗预训练大模型/rag/对话数据大全/{xuanzekeshi}.jsonl",
            'r', encoding='utf-8') as file:
        for line in file:
            # 解析每一行作为JSON对象
            json_object = json.loads(line)
            fangan_knowledge.append(json_object)
    try:
        recall_result = rag_recall(question, faiss_index, id_knowledge,llm_model, llm_tokenizer, recall_tokenizer, recall_model, rank_model, rank_tokenizer)
    except Exception as e:
        print(e)


    # 参考经验
    prompt = get_prompt(recall_result, fangan_knowledge)
    if prompt != []:
        instruction = prompt + f"\n参考上述实际案例，回答问题:{question}，请先一步步思考."
    return instruction


keshi_dist = [
    "泌尿外科",
    "产科",
    "儿内科",
    "耳鼻喉科",
    "妇产科",
    "肝胆外科",
    "感染科",
    "呼吸内科",
    "康复医学科",
    "口腔科",
    "内分泌科",
    "内科",
    "皮肤科",
    "皮肤性病科",
    "神经内科",
    "神经外科",
    "肾内科",
    "消化内科",
    "心理精神科",
    "心血管内科",
    "胸外科",
    "眼科",
    "整形外科",
    "中医科",
    "肿瘤内科"
]

def main():
    # torch.cuda.empty_cache()
    print('load model begin.')
    llm_model, llm_tokenizer, recall_tokenizer, recall_model, rank_model, rank_tokenizer = load_model()
    print('load model end.')
    # torch.cuda.empty_cache()
    print('load multimodel begin.')
    multimodel,multitokenizer = load_multimodel()
    print('load multimodel end.')

    user_avator = './user.png'
    robot_avator = './robot.png'

    st.title('XD_Docoter_GPT')

    generation_config = prepare_generation_config()


       # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message['role'], avatar=message.get('avatar')):
            st.markdown(message['content'])


    # 图片上传功能
    uploaded_image = st.file_uploader("上传图片", type=["png", "jpg", "jpeg"])
    image = None
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='上传的图片', use_column_width=True)
    #user_avatar= './robot.png'
    # Accept user input
    if prompt := st.chat_input('What is up?'):
        # Display user message in chat message container
        with st.chat_message('user', avatar=user_avator):
            st.markdown(prompt)
        print("用户输入问题：\n"+prompt)
        print("--------------------------------------")
        # Add user message to chat history
        st.session_state.messages.append({
            'role': 'user',
            'content': prompt,
            'avatar': user_avator
        })
        # 调用聊天逻辑处理用户输入
        text = chat_with_model(image,multimodel,multitokenizer)
        xunzhaokeshi = f"""你是一个专业的医生，你负责把用户的问题分配给相应的科室医生，请根据用户的问题判断这个问题和哪个科室类别最相关。
必须从以下科室列表中选择(并严格按照以下科室列表中的名字输出)。
用户问题：
{prompt}
{text}
科室列表：
{keshi_dist}
请按照以下格式输出：
说明理由：（解释选择科室理由）
科室选择：[（从上述科室列表中选择合适的科室）]
"""
        real_prompt = combine_history(xunzhaokeshi)
        print(xunzhaokeshi)
        print("-------------------------------------------")
        with st.chat_message('robot', avatar=robot_avator):
            message_placeholder = st.empty()
            response = ''
            for cur_response in generate_interactive(
                    model=llm_model,
                    tokenizer=llm_tokenizer,
                    prompt=real_prompt,
                    additional_eos_token_id=92542,
                    **asdict(generation_config),
            ):
                # Display robot response in chat message container
                message_placeholder.markdown(cur_response + '▌')
                


            print(f"模型分析问题并选择科室：\n{cur_response}")
            print("-----------------")
            message_placeholder.markdown(cur_response)
            response += cur_response
            print(response)
            xuanzekeshi = extract_keshi(response,llm_model, llm_tokenizer, recall_tokenizer, recall_model, rank_model, rank_tokenizer)
            st.code(f"将问题分配给{xuanzekeshi}医生")

            status_placeholder = st.empty()

            # 在占位符中显示“检索中”
            with status_placeholder.container():
                st.code("检索相关科室信息...")
            prompt=prompt+"\n"+text
            instruction = chongxie_zhaohui_jingpai(xuanzekeshi, prompt,llm_model, llm_tokenizer, recall_tokenizer, recall_model, rank_model, rank_tokenizer)

            # 在占位符中显示“检索完成”
            with status_placeholder.container():
                st.code("检索完成")
        real_instruction = combine_history2(instruction)
        print(instruction)
        with st.chat_message('robot', avatar=robot_avator):
            message_placeholder = st.empty()
            response=''
            for cur_response in generate_interactive(
                    model=llm_model,
                    tokenizer=llm_tokenizer,
                    prompt=real_instruction,
                    additional_eos_token_id=92542,
                    **asdict(generation_config),
            ):
                # Display robot response in chat message container
                message_placeholder.markdown(cur_response + '▌')
            message_placeholder.markdown(cur_response)
            response += cur_response
            print(response)
            print("-----------------")
        # Add robot response to chat history
        st.session_state.messages.append({
            'role': 'robot',
            'content': cur_response,  # pylint: disable=undefined-loop-variable
            'avatar': robot_avator,
        })

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

