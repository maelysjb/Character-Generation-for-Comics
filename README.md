# Generating Comic Characters with Generative AI

## About the Project
This is our final Master's thesis for the Data Science MSc program at the Barcelona School of Economics (BSE). The goal is to create variations of a new character designed by an artist using computational image generation. We collaborated with Jordane Meignaud (Instagram: @surunnuagecreation), who invented a character and provided six unique images of this character. All rights regarding the image belong to Jordane Meignaud, and any other uses of these images must be approved by her. Using these six images (or fewer) and various models, particularly Diffusion Models, we show that fine-tuning is the best method for addressing data scarcity and generating new poses for the character. 

<img src="https://github.com/maelysjb/Comics-GenAI/assets/145024696/84bb4ad8-85ec-4368-9ff7-6824e9acd587" width="100"> 
<img src="https://github.com/maelysjb/Comics-GenAI/assets/145024696/8d53ead8-1f7b-47f4-8b56-37c94f46992a" width="100"> 
<img src="https://github.com/maelysjb/Comics-GenAI/assets/145024696/8aba99b7-6f23-4868-a885-aa9b52604126" width="100"> 
<img src="https://github.com/maelysjb/Comics-GenAI/assets/145024696/d92a7656-eb92-4202-98f7-e3eb5b4a46e3" width="100">
<img src="https://github.com/maelysjb/Comics-GenAI/assets/145024696/0c79b782-9bbb-476b-9c97-e7e2810d150c" width="100">
<img src="https://github.com/maelysjb/Comics-GenAI/assets/145024696/1c81c33b-43bd-4c8e-8084-1609be731d9f" width="100">

## Contact the Authors
For any additional questions, feel free to reach out to the authors of this project:
* Maëlys Boudier (maelys.boudier@bse.eu)
* Natalia Beltrán (natalia.beltran@bse.eu)
* Arianna Michelangelo (arianna.michelangelo@bse.eu)


## General Set-Up

**Set-Up Model and Dataset on Hugging Face:** To facilitate saving model weights, we created an account on Hugging Face (which is free) and created a "New Model" with the name of our choice. Whenever we ran codes we could manually upload weights to this model space or sometimes directly save it in the hugging face directory as opposed to our local directory. We also set up a "New Dataset" with our training images, this is a useful alternative to using the images on the local directory as we tested many methods to run our codes (Google Colab, different Local Computers, Kaggle). Having a dataset on Hugging Face meant that we could also directly access our dataset without having to update file paths. 

**Secret Key on Hugging Face:** We chose to have private models and datasets on Hugging Face. As such, we were able to access them using a secret key called "Access Tokens" in the account parameters on Hugging Face. Make sure that you use a token of type "WRITE" if you intend on directly saving your models to the model directory on Hugging Face. 

*Note:* You can also save your keys in Kaggle in a hidden User Secrets and load them secretly so you avoid accidentally sharing your private secret keys.

```
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("HF_TOKEN")
``` 

**GPU on Kaggle:** 
After struggling with GPU usage, we decided to run our computationally demanding codes on Kaggle which enables each user to have 30 hours a week of GPU usage. We primarily used the P100 GPU which made our LoRA and Dream Booth codes over 10 times faster than when running on our laptops.

## Baseline Models

**GAN Model** 

We explored available literature and found limited research on comic character generation, particularly with scarce data from previous years. While Generative Adversarial Networks (GANs) have been used for image generation (refer to Marnix Verduyn's paper), they face significant challenges with less than 100k images. Specifically, the discriminator tends to overfit while the generator underfits (as discussed in the Nvidia blog paper).

1. **Comic Art Generation using GANs** by ir. Marnix Verduyn  
   Academic year 2021 – 2022  
   [Read the paper](https://www.nix.be/assets/pdf/Masterproef_MarnixVerduyn_KUL_MAI_2022.pdf)

2. **NVIDIA Research Achieves AI Training Breakthrough Using Limited Datasets**  
   December 7, 2020 by Isha Salian  
   [Read the blog](https://blogs.nvidia.com/blog/neurips-research-limited-data-gan/#:~:text=It%20typically%20takes%2050%2C000%20to,falter%20at%20producing%20realistic%20results.)

Our attempt to generate a baseline using a GAN with just 6 images demonstrated the inadequacy of this approach. The model failed to converge, highlighting the necessity for alternative methods or larger datasets.

<img src="https://github.com/maelysjb/Comics-GenAI/assets/145024696/4b4bae90-36ad-40e8-91ac-8be6219b89dc" width="100"> 

**Diffusion Model** 

<img src="https://github.com/maelysjb/Comics-GenAI/assets/145024696/5e3d19e8-046f-48b0-8f77-9f20f6ae904e" width="200"> 

**Stable Diffusion Model** 


## Fine Tuned Models

**DreamBooth Model** 

**LoRA Model** 

**DreamBooth + LoRA Model** 

<img src="https://github.com/maelysjb/Comics-GenAI/assets/145024696/f122b751-fa60-44c7-9b87-f0192e65b9f5" width="300"> 

## How to navigate the repository
```bash 
├── 1. Data
│   ├── sks-01.jpg
│   ├── sks-02.jpg
│   ├── sks-03.jpg
│   ├── sks-04.jpg
│   ├── sks-05.jpg
│   └── sks-06.jpg
├── 2. Descriptive Statistics
│   ├── Image_DataOverview.ipynb
│   ├── Image_ColorBreakdown.ipynb
│   └── Image_HSV.ipynb
├── 3. Baseline Models
│   ├── GAN Models
│   │   ├── GAN_Model_Version1.ipynb
│   │   └── GAN_Model_Version2.ipynb
│   ├── Diffusion Models
│   │   ├── Diffusion_32x32.ipynb
│   │   └── Diffusion_256x256.ipynb
│   └── Stable-Diffusion-XL-Prompt.ipynb
├── 4. Fine Tuning Models
│   ├── DreamBooth
│   │   ├── DreamBooth.ipynb
│   │   ├── DreamBooth_Inference.ipynb
│   │   └── DreamBooth_GoogleColab.ipynb
│   ├── LoRA
│   │   └──LoRA.ipynb
│   └── DreamBooth-LoRA
│   │   ├── DreamBooth-LoRA.ipynb
│   │   └── DreamBooth-LoRA-Inference.ipynb
├── 5. Documents
│   ├── Latex Report
│   │   └── Thesis_Masters_GENAI.tex
│   ├── Report.pdf
│   └── Presentation.pdf
├── 6. Generated Images
│   ├── DreamBooth
│   │   ├── DreamBooth200
│   │   ├── DreamBooth600
│   │   ├── DreamBooth800
│   │   ├── DreamBooth1000
│   │   ├── DreamBooth1500
│   │   └── DreamBooth2000
│   ├── DreamBooth-LoRA
│   │   ├── 100_steps
│   │   ├── 500_steps
│   │   ├── 800_steps
│   │   ├── 1000_steps
│   │   ├── 1200_steps
│   │   ├── 1500_steps
│   │   ├── 1800_steps
│   │   ├── 2000_steps
│   │   └── Gif_outputs
│   └──  LoRA
└── README.md
```
