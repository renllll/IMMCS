# IMMCS
本项目全称为视音文结合的多级科室医疗问诊系统，旨在打造一个完整、真实、可信的医疗问诊系统。该系统综合考虑用户提交的医疗图片和问诊问题，通过专业的医疗大模型进行分级科室的任务分配。目前，分级科室的知识库由爬取自千问健康的各科室医疗数据构建，医疗大模型通过这些知识库扮演不同科室的医生。通过分科室诊疗的方法，能大幅降低数据冲突的问题，提供更精确、更详细的回复。未来，我们会为每个科室设计不同的工具链，以获得更精确的医疗诊断。

系统主要构成如下：
1.医疗多模态模型：由我们搜集的公开医疗数据集和自建医疗数据集训练而成。
2.医疗大模型：目前使用的是internlm2-7b-chat模型，后续我们将开源经过增量预训练和全量微调的自研医疗模型。
3.Recall和Rank模型：我们使用GPT-4生成的医疗科室分类数据集，并用此数据集训练了医疗分类模型。该模型对原始爬取的医疗数据进行分类，分类后的数据集用于构建知识库，并训练Recall和Rank模型。
其流程图如下：
<img width="623" alt="image" src="https://github.com/renllll/IMMCS/assets/103827697/ff050c80-ad5b-40b8-a318-7cd4d1185b72">
总体流程图如下：
<img width="821" alt="1719121372317" src="https://github.com/renllll/IMMCS/assets/103827697/9860999f-2535-4082-91fd-d41a50c5f44a">
