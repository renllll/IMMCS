# IMMCS
本项目全称为视音文结合的多级科室医疗问诊系统，旨在打造一个完整、真实、可信的医疗问诊系统。该系统综合考虑用户提交的医疗图片和问诊问题，通过专业的医疗大模型进行分级科室的任务分配。目前，分级科室的知识库由爬取自千问健康的各科室医疗数据构建，医疗大模型通过这些知识库扮演不同科室的医生。通过分科室诊疗的方法，能大幅降低数据冲突的问题，提供更精确、更详细的回复。未来，我们会为每个科室设计不同的工具链，以获得更精确的医疗诊断。

医疗分科室知识库的模型和数据存储地址为https://openxlab.org.cn/models/detail/yuanyizhixie/Medical_department_knowledge_base

医疗多模态模型存储地址为https://openxlab.org.cn/models/detail/yuanyizhixie/Medical_multi_modal_large_model

演示视频地址为： https://www.bilibili.com/video/BV1Bz3gejEuv/

demo网站（暂时）： https://bdbd-185-248-184-99.ngrok-free.app

系统主要构成如下：

1.医疗多模态模型：由我们搜集的公开医疗数据集和自建医疗数据集训练InternLM-XComposer2_vl_7b而成。

2.医疗大模型：目前使用的是internlm2-7b-chat模型，后续我们将开源经过增量预训练和全量微调的自研医疗模型。

3.Recall和Rank模型：我们使用GPT-4生成的医疗科室分类数据集，并用此数据集训练了医疗分类模型。该模型对原始爬取的医疗数据进行分类，分类后的数据集用于构建知识库，并训练Recall和Rank模型。


其流程图如下：


<img width="623" alt="image" src="https://github.com/renllll/IMMCS/assets/103827697/ff050c80-ad5b-40b8-a318-7cd4d1185b72">





总体流程图如下：




<img width="831" alt="1719123974127" src="https://github.com/renllll/IMMCS/assets/103827697/42a83d57-045e-424a-8671-32e303432aef">




**环境搭建**
```
git clone https://github.com/renllll/IMMCS/edit/main

conda create -n  docoter python==3.10

pip install -r requirements.txt
```
**下载模型**
```
git clone https://code.openxlab.org.cn/yuanyizhixie/Medical_multi_modal_large_model.git

git clone https://code.openxlab.org.cn/yuanyizhixie/Medical_department_knowledge_base.git
```
请将web1或者xitong文件里所有模型地址修改为本地保存地址


**模型部署**

分科室诊疗系统模型部署

请在修改web1文件的模型地址后运行
```
streamlit run web1.py
```
默认在卡1部署，需要24gb的显存

医疗多模态模型部署
```
python multi_web.py
```
视音文结合的分级科室问诊系统部署
```
streamlit run  xitong.py
```
注意这需要两张3090或者一张a100能完全部署，需要至少40gb的显存
