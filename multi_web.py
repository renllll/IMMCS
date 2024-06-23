import os
import sys
import argparse
import torch
from transformers import AutoTokenizer,AutoModel
from peft import AutoPeftModelForCausalLM
import gradio as gr
from PIL import Image
from transformers import AutoModelForCausalLM


# 禁用梯度计算
torch.set_grad_enabled(False)

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--dtype", default='fp16', type=str)
parser.add_argument("--port", default=11111, type=int)
args = parser.parse_args()

# 初始化模型和分词器
model_path = "path_to_multimodel"

model = AutoModel.from_pretrained(
    model_path,
   # device_map='auto',
    torch_dtype=torch.float16,
    trust_remote_code=True
).eval().cuda()
if args.num_gpus > 1:
    from accelerate import dispatch_model
    device_map = auto_configure_device_map(args.num_gpus)
    model = dispatch_model(model, device_map=device_map)



tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 定义存储图片的文件夹
image_dir = "images"
os.makedirs(image_dir, exist_ok=True)

# 定义聊天逻辑
def chat_with_model(image, text, history):
    if image is not None:
        image_path = os.path.join(image_dir, "uploaded_image.png")
        image.save(image_path)
    else:
        image_path = "1.png"  # 使用默认图像路径
    text="<ImageHere> "+text
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            response, _ = model.chat(tokenizer, query=text, image=image_path, history=history, do_sample=False)

    history.append((text, response))
    return history, history

def display_image(image):
    return image

def clear_history():
    if os.path.exists(image_dir):
        for filename in os.listdir(image_dir):
            file_path = os.path.join(image_dir, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    return [], []

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("# 图像描述模型")
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="上传图片")
            text_input = gr.Textbox(label="输入文本")
            chat_history = gr.Chatbot(label="对话历史")
            submit_btn = gr.Button("发送")
            clear_btn = gr.Button("清空历史")
        with gr.Column():
            img_output = gr.Image(label="上传的图片")

    submit_btn.click(chat_with_model, [img_input, text_input, chat_history], [chat_history, chat_history])
    img_input.change(display_image, inputs=img_input, outputs=img_output)
    clear_btn.click(clear_history, outputs=[chat_history, chat_history])

demo.launch(server_name="0.0.0.0", server_port=args.port)


