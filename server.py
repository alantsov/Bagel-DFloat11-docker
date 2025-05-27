# flask server

import os
import torch
import random
import numpy as np
from io import BytesIO
from flask import Flask, request, send_file, jsonify

from accelerate import infer_auto_device_map, dispatch_model, init_empty_weights
from PIL import Image

from data.data_utils import add_special_tokens
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer

from dfloat11 import DFloat11Model

app = Flask(__name__)

# Set seed function from app.py
def set_seed(seed):
    """Set random seeds for reproducibility"""
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed

# Load model weights before starting the server
print("Loading model weights...")

# Model Initialization (same as in app.py)
model_path = "./BAGEL-7B-MoT-DF11"

llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers -= 1

vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "vae/ae.safetensors"))

config = BagelConfig(
    visual_gen=True,
    visual_und=True,
    llm_config=llm_config, 
    vit_config=vit_config,
    vae_config=vae_config,
    vit_max_num_patch_per_side=70,
    connector_act='gelu_pytorch_tanh',
    latent_patch_size=2,
    max_latent_size=64,
)

with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)

model = model.to(torch.bfloat16)
model.load_state_dict({
    name: torch.empty(param.shape, dtype=param.dtype, device='cpu') if param.device.type == 'meta' else param
    for name, param in model.state_dict().items()
}, assign=True)

DFloat11Model.from_pretrained(
    model_path,
    bfloat16_model=model,
    device='cpu',
)

# Model Loading and Multi GPU Inference Preparing
device_map = infer_auto_device_map(
    model,
    max_memory={0: "24GiB"},
    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer", "SiglipVisionModel"],
)

same_device_modules = [
    'language_model.model.embed_tokens',
    'time_embedder',
    'latent_pos_embed',
    'vae2llm',
    'llm2vae',
    'connector',
    'vit_pos_embed'
]

if torch.cuda.device_count() == 1:
    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device
        else:
            device_map[k] = "cuda:0"
else:
    first_device = device_map.get(same_device_modules[0])
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device

model = dispatch_model(model, device_map=device_map, force_hooks=True)
model = model.eval()

# Inferencer Preparing 
inferencer = InterleaveInferencer(
    model=model,
    vae_model=vae_model,
    tokenizer=tokenizer,
    vae_transform=vae_transform,
    vit_transform=vit_transform,
    new_token_ids=new_token_ids,
)

print("Model loaded successfully!")

# Endpoint 1: ping (no parameters)
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "ok"})

# Endpoint 2: generate (prompt)
@app.route('/generate', methods=['GET'])
def generate():
    # Get prompt as a mandatory query parameter
    prompt = request.args.get('prompt')
    if not prompt:
        return jsonify({"error": "Prompt parameter is required"}), 400

    # Get optional parameters with default values from app.py
    show_thinking = request.args.get('show_thinking', 'false').lower() == 'true'
    cfg_text_scale = float(request.args.get('cfg_text_scale', 4.0))
    cfg_interval = float(request.args.get('cfg_interval', 0.4))
    timestep_shift = float(request.args.get('timestep_shift', 3.0))
    num_timesteps = int(request.args.get('num_timesteps', 50))
    cfg_renorm_min = float(request.args.get('cfg_renorm_min', 1.0))
    cfg_renorm_type = request.args.get('cfg_renorm_type', 'global')
    max_think_token_n = int(request.args.get('max_think_token_n', 1024))
    do_sample = request.args.get('do_sample', 'false').lower() == 'true'
    text_temperature = float(request.args.get('text_temperature', 0.3))
    seed = int(request.args.get('seed', 0))
    image_ratio = request.args.get('image_ratio', '1:1')

    # Set seed for reproducibility
    set_seed(seed)

    # Determine image shapes based on image_ratio
    if image_ratio == "1:1":
        image_shapes = (1024, 1024)
    elif image_ratio == "4:3":
        image_shapes = (768, 1024)
    elif image_ratio == "3:4":
        image_shapes = (1024, 768) 
    elif image_ratio == "16:9":
        image_shapes = (576, 1024)
    elif image_ratio == "9:16":
        image_shapes = (1024, 576)
    else:
        image_shapes = (1024, 1024)  # Default to 1:1

    # Set hyperparameters
    inference_hyper = dict(
        max_think_token_n=max_think_token_n if show_thinking else 1024,
        do_sample=do_sample if show_thinking else False,
        text_temperature=text_temperature if show_thinking else 0.3,
        cfg_text_scale=cfg_text_scale,
        cfg_interval=[cfg_interval, 1.0],  # End fixed at 1.0
        timestep_shift=timestep_shift,
        num_timesteps=num_timesteps,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
        image_shapes=image_shapes,
    )

    # Call inferencer with or without think parameter based on user choice
    result = inferencer(text=prompt, think=show_thinking, **inference_hyper)

    # Convert PIL image to bytes
    img_byte_arr = BytesIO()
    result["image"].save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Return the PNG file
    return send_file(img_byte_arr, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
