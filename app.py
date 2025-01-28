import gradio as gr
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from PIL import Image
import numpy as np
import os
import time
from Upsample import RealESRGAN
import random

# Load model and processor
model_path = "deepseek-ai/Janus-Pro-7B"
config = AutoConfig.from_pretrained(model_path)
language_config = config.language_config
language_config._attn_implementation = 'eager'
vl_gpt = AutoModelForCausalLM.from_pretrained(model_path,
                                             language_config=language_config,
                                             trust_remote_code=True)
if torch.cuda.is_available():
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda()
else:
    vl_gpt = vl_gpt.to(torch.float16)

vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
cuda_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# SR model
sr_model = RealESRGAN(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), scale=2)
sr_model.load_weights(f'weights/RealESRGAN_x2.pth', download=False)

@torch.inference_mode()
def multimodal_understanding(image, question, seed, top_p, temperature):
    # Clear CUDA cache before generating
    torch.cuda.empty_cache()
    
    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    
    pil_images = [Image.fromarray(image)]
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(cuda_device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16)
    
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False if temperature == 0 else True,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
    )
    
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer

def generate(input_ids,
             width,
             height,
             temperature: float = 1,
             parallel_size: int = 5,
             cfg_weight: float = 5,
             image_token_num_per_image: int = 576,
             patch_size: int = 16):
    # Clear CUDA cache before generating
    torch.cuda.empty_cache()
    
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).to(cuda_device)
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id
    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(cuda_device)

    pkv = None
    for i in range(image_token_num_per_image):
        with torch.no_grad():
            outputs = vl_gpt.language_model.model(inputs_embeds=inputs_embeds,
                                                use_cache=True,
                                                past_key_values=pkv)
            pkv = outputs.past_key_values
            hidden_states = outputs.last_hidden_state
            logits = vl_gpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)

            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

    patches = vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int),
                                                 shape=[parallel_size, 8, width // patch_size, height // patch_size])

    return generated_tokens.to(dtype=torch.int), patches

def unpack(dec, width, height, parallel_size=5):
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, width, height, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    return visual_img

@torch.inference_mode()
def generate_image(prompt,
                   seed=None,
                   guidance=5,
                   t2i_temperature=1.0,
                   num_images=5):
    # Clear CUDA cache and avoid tracking gradients
    torch.cuda.empty_cache()
    
    # Set the seed for reproducible results
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
    
    width = height = 384  # Fixed square dimensions
    parallel_size = num_images
    
    with torch.no_grad():
        messages = [{'role': '<|User|>', 'content': prompt},
                    {'role': '<|Assistant|>', 'content': ''}]
        text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(conversations=messages,
                                                                   sft_format=vl_chat_processor.sft_format,
                                                                   system_prompt='')
        text = text + vl_chat_processor.image_start_tag
        
        input_ids = torch.LongTensor(tokenizer.encode(text))
        output, patches = generate(input_ids,
                                   width,
                                   height,
                                   cfg_weight=guidance,
                                   parallel_size=parallel_size,
                                   temperature=t2i_temperature)
        images = unpack(patches,
                        width,
                        height,
                        parallel_size=parallel_size)

        stime = time.time()
        ret_images = [Image.fromarray(images[i]) for i in range(parallel_size)]
        print(f'processing time: {time.time() - stime}')
        return ret_images

def image_upsample(img: Image.Image) -> Image.Image:
    if img is None:
        raise Exception("Image not uploaded")
    
    width, height = img.size
    
    if width >= 5000 or height >= 5000:
        raise Exception("The image is too large.")

    result = sr_model.predict(img.convert('RGB'))
    return result

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(value="# Multimodal Understanding")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="Upload Image")
        
        with gr.Column():
            question_input = gr.Textbox(
                label="Question",
                placeholder="Ask a question about the image..."
            )
            with gr.Row():
                und_seed_input = gr.Number(
                    label="Seed",
                    precision=0,
                    value=42
                )
                random_seed_button = gr.Button("Random Seed")
            top_p = gr.Slider(
                minimum=0,
                maximum=1,
                value=0.95,
                step=0.05,
                label="Top-p"
            )
            temperature = gr.Slider(
                minimum=0,
                maximum=1,
                value=0.1,
                step=0.05,
                label="Temperature"
            )
    
    understanding_button = gr.Button("Ask Question")
    understanding_output = gr.Textbox(
        label="Response",
        placeholder="AI response will appear here..."
    )

    examples_understanding = gr.Examples(
        examples=[
            ["Explain this meme", "doge.png"],
            ["Convert the formula into latex code", "equation.png"]
        ],
        inputs=[question_input, image_input]
    )

    gr.Markdown(value="# Text-to-Image Generation +")
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Describe the image you want to generate..."
            )
            with gr.Row():
                seed_input = gr.Number(
                    label="Seed (Optional)",
                    precision=0,
                    value=1234
                )
                t2i_random_seed_button = gr.Button("Random Seed")
            with gr.Row():
                cfg_weight_input = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=0.5,
                    label="CFG Weight"
                )
                t2i_temperature = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=1.0,
                    step=0.05,
                    label="Temperature"
                )
            
            with gr.Row():
                num_images_input = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Number of Images"
                )
            
            generation_button = gr.Button("Generate Images")
    
    image_output = gr.Gallery(
        label="Generated Images",
        columns=3,
        rows=None,
        height=800
    )

    examples_t2i = gr.Examples(
        label="Text to image generation examples",
        examples=[
            "Master shifu racoon wearing drip attire as a street gangster.",
            "The face of a beautiful girl",
            "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
            "A cute and adorable baby fox with big brown eyes, autumn leaves in the background",
            "An intricately designed eye set against a circular backdrop with ornate swirl patterns"
        ],
        inputs=prompt_input
    )

    # Event handlers
    def generate_random_seed():
        return random.randint(0, 2**32 - 1)

    random_seed_button.click(
        fn=generate_random_seed,
        outputs=und_seed_input
    )

    t2i_random_seed_button.click(
        fn=generate_random_seed,
        outputs=seed_input
    )

    understanding_button.click(
        fn=multimodal_understanding,
        inputs=[
            image_input,
            question_input,
            und_seed_input,
            top_p,
            temperature
        ],
        outputs=understanding_output
    )
    
    generation_button.click(
        fn=generate_image,
        inputs=[
            prompt_input,
            seed_input,
            cfg_weight_input,
            t2i_temperature,
            num_images_input
        ],
        outputs=image_output
    )

demo.launch(share=True)