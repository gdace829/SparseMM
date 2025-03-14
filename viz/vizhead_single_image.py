from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
import numpy as np
import torch
import sys
import base64
from PIL import Image
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import os

sys.path.append("./visual_head/LLaVA-NeXT")
try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
    from llava.conversation import conv_templates, SeparatorStyle
except ImportError:
    print("LLaVA is not installed. Please install LLaVA to use this model.")

pretrained = "/path/to/models/llava-v1.6"
enc, model, image_processor, max_length = load_pretrained_model(pretrained, None, get_model_name_from_path(pretrained), attn_implementation='eager')
config = model.config
config = config
conv_mode = "vicuna_v1"

layer_num, head_num = config.num_hidden_layers, config.num_attention_heads
print(f"layer number: {layer_num}, head number {head_num}")


def cusum_attention(score, attention, prompt_len, image_feature_len):
    # 将原有的分数数组转换为 PyTorch 张量
    score_tensor = torch.tensor(score, dtype=torch.float32)  # 保证数据类型一致
    
    # 获取 attention 的相关形状
    layer_num = len(attention)
    head_num = len(attention[0][0])
    
    for layer_idx in range(layer_num):
        # 提取这一层的 attention 张量
        layer_attention = attention[layer_idx][0]  # 拿出 batch_size 为0的情况
        
        # 提取 prompt 之后的 image feature 部分
        # import pdb; pdb.set_trace()
        values = layer_attention[:, 0, prompt_len: prompt_len + image_feature_len].to('cpu')  # 选择第一个head

        # 累加到对应的 score
        score_tensor[layer_idx] += values

    return score_tensor.tolist()  # 转换回列表


def decode(outputs, inp, prompt_len, max_decode_len, image_feature_len, block_list=None):
    output = []
    prompt_score = [[[0 for _ in range(image_feature_len)] for _ in range(head_num)] for _ in range(layer_num)]
    past_kv = outputs.past_key_values
    for step_i in range(max_decode_len):
        inp = inp.view(1, 1)
        outputs, _ = model(input_ids=inp, past_key_values=past_kv, use_cache=True, output_attentions=True)
        past_kv = outputs.past_key_values
        inp = outputs.logits[0, -1].argmax()
        step_token = enc.convert_ids_to_tokens(inp.item())
        # step_token = enc.decode(inp.item())
        print(step_token)
        output.append(inp.item())
        if outputs.attentions[0] is not None:
            prompt_score = cusum_attention(prompt_score, outputs.attentions, prompt_len, image_feature_len)

        if inp.item()==2:  # end of sentence
            break

    return output, prompt_score


def process_heat_map(prompt_score, return_image_feature, image_size):
    '''
    prompt_score: list of scores corresponding to each patch
    return_image_feature: (channel, height, width)
    image_size: (width, height) of the original image
    '''
    # Initialize a zero matrix with the same size as the feature map
    score_matrix = torch.zeros(return_image_feature.shape[1], return_image_feature.shape[2])

    origin_img_width, origin_img_height = image_size
    num_patches_width, num_patches_height = return_image_feature.shape[2], return_image_feature.shape[1]
    
    # Compute the dimensions of each patch in the original image
    mapping_pixel_width = origin_img_width // num_patches_width
    mapping_pixel_height = origin_img_height // num_patches_height
    # import pdb; pdb.set_trace()
    # Iterate over patches and assign scores
    idx = 0
    for i in range(num_patches_height):
        for j in range(num_patches_width + 1):
            if j < num_patches_width:
                score_matrix[i, j] = prompt_score[idx]
            idx += 1
    # import pdb; pdb.set_trace()
    min_val = score_matrix.min()
    max_val = score_matrix.max()
    # import pdb; pdb.set_trace()
    print(f"Score matrix min value: {min_val}, max value: {max_val}")

    if max_val > min_val:  # Avoid division by zero
        normalized_matrix = (score_matrix - min_val) / (max_val - min_val)
    else:
        normalized_matrix = score_matrix

    # Convert score matrix to numpy array if needed, and return
    return normalized_matrix.cpu().numpy()

def draw(prompt_score, image_path, save_path, return_image_feature, image_size):
    ind = [(i, j) for i in range(layer_num) for j in range(head_num)]

    for (layer_idx, head_idx) in ind:
        # 1. 加载背景图片
        background = cv2.imread(image_path)
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)  # 转为 RGB 格式

        # 2. 生成热力图
        # 将 prompt_score 转为 numpy 数组
        heatmap_data = process_heat_map(prompt_score[layer_idx][head_idx], return_image_feature, image_size)
        
        # 设置热力图风格
        plt.figure(figsize=(10, 10))
        sns.heatmap(heatmap_data, cmap='Reds', cbar=False)
        plt.axis('off')

        # 保存热力图为临时文件
        heatmap_path = f"temp_heatmap.png"
        plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # 3. 加载热力图并调整大小
        heatmap = cv2.imread(heatmap_path)
        heatmap = cv2.resize(heatmap, (background.shape[1], background.shape[0]))
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        heatmap_hsv = cv2.cvtColor(heatmap, cv2.COLOR_RGB2HSV)
        heatmap_hsv[:,:,1] = np.clip(heatmap_hsv[:,:,1] * 1.2, 0, 255)  # 增加饱和度
        heatmap = cv2.cvtColor(heatmap_hsv, cv2.COLOR_HSV2RGB)

        # 4. 图像叠加
        blended = cv2.addWeighted(background, 0.5, heatmap, 0.5, 0)  # 调整透明度

        # 5. 保存结果
        result = Image.fromarray(blended)
        result.save(save_path+'layer_{}_head_{}.png'.format(layer_idx, head_idx))
        print(f"layer_{layer_idx}_head_{head_idx} 热力图保存至: {save_path}")
            

def generate_func(image_name):
    image_path = './viz/images/' + image_name
    save_path = './viz/single_image_tmp/' + image_name.replace('.jpg', '') + '/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Directory '{save_path}' created.")

    image = Image.open(image_path).convert("RGB")
    image_size = image.size

    image_tensor = process_images([image], image_processor, config)
    # import pdb;pdb.set_trace()

    # question = "Provide all the OCR results of this image."
    question = "Describe the object in this image."
    question = DEFAULT_IMAGE_TOKEN + '\n' + question
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, enc, IMAGE_TOKEN_INDEX, return_tensors='pt')
    # import pdb; pdb.set_trace()
    prompt_len = len(input_ids) - 1
    input_ids = input_ids.to(device='cuda', non_blocking=True).unsqueeze(0)
    input_ids = input_ids.to(model.device)
        
    with torch.no_grad():
        outputs, return_image_feature = model(
            input_ids = input_ids[:,:-1],
            images = image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
            image_sizes = [image_size], 
            use_cache=True, return_dict=True
        )
        # print(ocr_idxs)
        # import pdb; pdb.set_trace()
        prompt_len += 576 # base image for llava model
        prefix_len = int(torch.where(input_ids[0] < 0)[0][0]) + 576

        image_feature_len = outputs.past_key_values[0][0].shape[2] - prompt_len 
        output, prompt_score = decode(outputs, input_ids[:,-1], prefix_len, max_decode_len=128,image_feature_len=image_feature_len)
        response = enc.decode(output,skip_special_tokens=True).strip()

        draw(prompt_score, image_path, save_path, return_image_feature, image_size)
        print("output:", response)



if __name__ == "__main__":
    generate_func('book.jpg')
