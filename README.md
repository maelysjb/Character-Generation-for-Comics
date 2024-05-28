# Generating Comic Characters with Generative AI

## About the Project
This is our final Master's thesis for the Data Science MSc program at the Barcelona School of Economics (BSE). The goal is to create variations of a new character designed by an artist using computational image generation. We collaborated with Jordane Meignaud (Instagram: @surunnuagecreation), who invented a character and provided six unique images of this character. All rights regarding the image belong to Jordane Meignaud, and any other uses of these images must be approved by her. Using these six images (or fewer) and various models, particularly Diffusion Models, we show that fine-tuning is the best method for addressing data scarcity and generating new poses for the character. 

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
After struggling with GPU usage, we decided to run our computationally demanding codes on Kaggle which enables each user to ahve 30 hours a week of GPU usage. We primarily used the P100 GPU which made our LoRA and Dream Booth codes over 10 times faster than on our laptops.

## Baseline Models

**GAN Model** 

**Diffusion Model** 

**Stable Diffusion Model** 


## Fine Tuned Models

**DreamBooth Model** 

**LoRA Model** 

**DreamBooth + LoRA Model** 


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
│   ├── Nat
│   └── Ari
├── 3. Baseline Models
│   ├── GAN Models
│   │   ├── GAN_Model_Version1.ipynb
│   │   └── GAN_Model_Version2.ipynb
│   ├── Diffusion Models
│   │   ├── Diffusion_32x32.ipynb
│   │   └── Diffusion_256x256.ipynb
│   └── Stable-Diffusion-XL-Prompt.ipynb
├── 4. Fine Tuning Models
│   ├── DreamBooth.ipynb
│   ├── LoRA.ipynb
│   └── DreamBooth-LoRA.ipynb
├── 5. Documents
│   ├── Latex Report
│   │   └── Thesis_Masters_GENAI.tex
│   ├── Report.pdf
│   └── Presentation.pdf
└── README.md
```
