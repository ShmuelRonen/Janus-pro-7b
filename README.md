# Janus-Pro-7B: Multimodal Understanding and Text-to-Image Generation

**Janus-Pro-7B** is a powerful multimodal AI system that combines **image understanding** and **text-to-image generation** capabilities. Built on state-of-the-art transformer models, Janus-Pro-7B allows users to interact with images and generate high-quality visuals from textual descriptions. This project leverages deep learning frameworks like PyTorch and Hugging Face Transformers, providing a seamless interface for multimodal AI tasks.

![image](https://github.com/user-attachments/assets/e2d06aee-2227-4951-82d2-1dc6eda28ff1)

![image](https://github.com/user-attachments/assets/bc0b80f6-0efa-46a3-8fc6-fab9ceade5a9)

## Features

- **Multimodal Understanding**: Ask questions about images and receive detailed, context-aware responses.
- **Text-to-Image Generation**: Generate high-resolution images from textual prompts with customizable parameters like seed, temperature, and guidance scale.
- **Image Upscaling**: Enhance image resolution using advanced super-resolution techniques.
- **User-Friendly Interface**: Built with Gradio, Janus-Pro-7B offers an intuitive web-based interface for easy interaction.
- **Customizable Parameters**: Control generation settings such as seed, temperature, and top-p for tailored outputs.

## Key Technologies

- **PyTorch**: For deep learning model inference and training.
- **Hugging Face Transformers**: Leveraging pre-trained language and vision models.
- **Gradio**: For building interactive web interfaces.
- **RealESRGAN**: For high-quality image upscaling.

## Use Cases

- **Image Understanding**: Analyze and interpret images, answer questions, and extract insights.
- **Creative Image Generation**: Generate unique visuals from text prompts for art, design, and storytelling.
- **Educational Tools**: Assist in explaining visual content or generating illustrative materials.
- **Research and Development**: Experiment with multimodal AI models and generation techniques.

## Getting Started

### Installation

#### Option 1: Conda Environment (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/ShmuelRonen/janus-pro-7b.git
   cd janus-pro-7b
   ```

2. Create a Conda environment:
   ```bash
   conda create -n janus_pro_7b_env python=3.9
   conda activate janus_pro_7b_env
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the `RealESRGAN_x2.pth` model weights:
   - Visit the Hugging Face model page: [RealESRGAN_x2.pth](https://huggingface.co/ai-forever/Real-ESRGAN/blob/main/RealESRGAN_x2.pth).
   - Download the `RealESRGAN_x2.pth` file.
   - Place the downloaded file in the `weights` folder inside the project directory.
     
5. The pre-trained model weights will download automatic in first run.

### Running the Application

1. Launch the Gradio interface:
   ```bash
   python app.py
   ```

2. Open the provided URL in your browser to interact with the Janus-Pro-7B system.

## Examples

### Multimodal Understanding
- **Input**: Upload an image and ask, "What is happening in this scene?"
- **Output**: The model provides a detailed description of the image.

### Text-to-Image Generation
- **Input**: "A futuristic cityscape at sunset, cyberpunk style."
- **Output**: A set of high-quality generated images matching the description.

## Contributing

Contributions are welcome! If you'd like to improve Janus-Pro-7B, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **DeepSeek AI** for the foundational models.
- **Hugging Face** for the Transformers library.
- **Gradio** for the interactive interface framework.

---

This version replaces "Janus" with "Janus-Pro-7B" throughout the description and updates the repository name and Conda environment name accordingly. It maintains all the functionality and clarity of the original description while reflecting the updated project name.
