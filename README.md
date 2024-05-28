# Generating Comic Characters with Generative AI

## About the Project
This is our final Master's thesis for the Data Science MSc program at the Barcelona School of Economics (BSE). The goal is to create variations of a new character designed by an artist using computational image generation. We collaborated with Jordane Meignaud (Instagram: @surunnuagecreation), who invented a character and provided six unique images of this character. All rights regarding the image belong to Jordane Meignaud, and any other uses of these images must be approved by her. Using these six images (or fewer) and various models, particularly Diffusion Models, we show that fine-tuning is the best method for addressing data scarcity and generating new poses for the character. 

## Contact the Authors
For any additional questions, feel free to reach out to the authors of this project:
* Maëlys Boudier (maelys.boudier@bse.eu)
* Natalia Beltrán (natalia.beltran@bse.eu)
* Arianna Michelangelo (arianna.michelangelo@bse.eu)


## General Set-Up

**Set-Up Model on Hugging Face** 

**Set-Up Dataset on Hugging Face** 

**Secret Key on Hugging Face** 

**GPU on Kaggle** 


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
