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

<img width="450" alt="GAN_graph" src="https://github.com/maelysjb/Comics-GenAI/assets/145024696/0a9c7657-b7d2-464a-aa6e-0ca24300ba40">
<img src="https://github.com/maelysjb/Comics-GenAI/assets/145024696/4b4bae90-36ad-40e8-91ac-8be6219b89dc" width="200"> 

**Diffusion Model** 

Our next attempt at establishing a baseline involved training a full diffusion model, which operates through a noising and denoising process. While this approach also failed to achieve conclusive results, it occasionally generated a few blue pixels similar to the colors in our training images. This indicated some potential but reamined inconclusive.

<img width="450" src="https://github.com/maelysjb/Comics-GenAI/assets/145024696/516cbe72-f2d0-42c6-b633-be366272de12">
<img width="150" src="https://github.com/maelysjb/Comics-GenAI/assets/145024696/a7d07244-a28e-401f-94e1-0cede2fc0c28">

 

**Stable Diffusion Model** 

Our final attempt at establishing a baseline involved using a fully trained Stable Diffusion Model with a few carefully crafted prompts to approximate our training images. This approach allowed us to evaluate the potential visual quality we could achieve, even though the generated images did not perfectly capture all the specific features of our training data.

<img width="200" src="https://github.com/maelysjb/Comics-GenAI/assets/145024696/46f8bdc6-37cc-4568-a15e-79f5f4bb2491">



## Fine Tuned Models

**DreamBooth Model** 

We began by implementing the DreamBooth training technique which enables us to teach a new concept to a Stable Diffusion model through fine-tuning. This method entails adjusting the weights of a complete diffusion model while training it on a small set of images alongside a text embedding. Essentially, the method operates by converting prompts into text embeddings, introducing noise to the images, and directing the model to denoise them based on the provided concept. Through an iterative refinement process, the model's structure is honed until it effectively grasps the association. Ultimately, this enables the model to recognize and link the unique identifier “UnicornGirl” from the prompt with the associated image data.

<img width="400" alt="Dream_graph" src="https://github.com/maelysjb/Comics-GenAI/assets/145024696/984989cf-06d7-4714-b771-adb4f08c0db0">
<img width="270" alt="DREAMBOOTH_Image_Generated" src="https://github.com/maelysjb/Comics-GenAI/assets/145024696/bcfc8ffd-d137-4d7c-8637-00fa312e576e">

**LoRA Model** 

Additionally, we implemented the Low-Rank Adaptation Technique (LoRA), which was developed to address the challenge of fine-tuning large language models. When applied in the context of Stable Diffusion, this technique focuses on adapting only certain parts of the neural network. LoRA gets applied to the cross-attention layers that link our image data with the textual prompts. This allows our diffusion model to recognize new words as distinct concepts, enhancing its performance without altering its underlying structure and existing knowledge, and without the need to retrain all the weights each time. 

<img width="400" alt="LoRA_image" src="https://github.com/maelysjb/Comics-GenAI/assets/145024696/4ff2de04-357a-4b2f-85e9-21f4b3474c4d">
<img width="250" alt="LoRA_Generated_Imgae" src="https://github.com/maelysjb/Comics-GenAI/assets/145024696/878889e8-7544-4846-ba44-5bb7cd5e44b0">


**DreamBooth + LoRA Model** 

Lastly, we implemented a DreamBooth with LoRA fine-tuning, which offers notable advantages by incorporating additional trainable layers to the DreamBooth model without altering the original weights. During the fine-tuning process, both DreamBooth and LoRA weights are iteratively adjusted to better align with the targeted concept. DreamBooth weights are refined to enhance the model's capacity in associating the concept with the provided prompt and image data. Meanwhile, the LoRA weights are utilized to selectively adjust the significance of various features within the model, enabling it to focus more effectively on the nuances of the specific concept. Through this combined training process, the model progressively improves its ability to denoise images and associate the unique identifier with the represented concept. 

<img width="400" alt="dreamLora_graph" src="https://github.com/maelysjb/Comics-GenAI/assets/145024696/2e703606-14c7-4e91-8a5f-2a5e801b6339">
<img width="250" alt="DreamBoothLora_genimg" src="https://github.com/maelysjb/Comics-GenAI/assets/145024696/b749da27-ad73-4c71-a927-89dd00ca36b3">  


## Tuning the DreamBooth + LoRA Model 

**Training Steps** 

<img src="https://github.com/maelysjb/Comics-GenAI/assets/145024696/f122b751-fa60-44c7-9b87-f0192e65b9f5" width="300"> 

**Learning Rate** 

<img src="https://github.com/maelysjb/Comics-GenAI/assets/143039813/22ec201c-2679-463d-b942-c4810a4489f9" width="300"> 

**Inference Steps** 

<img src="https://github.com/maelysjb/Comics-GenAI/assets/143039813/03c84ba0-dc41-490d-882e-321f68169576" width="300"> 



For further information on the intricacies of any of the above techniques and findings, we invite you to read our comprehensive report. It provides an in-depth examination of our thesis real world applications, detailed explanations of tuning methods, and extensive insights into our findings! Access our full report here:  [5. Documents ](https://github.com/maelysjb/Comics-GenAI/blob/main/5.%20Documents/Latex%20Report/Thesis_Masters_GENAI.pdf)

## How to navigate the repository
```bash 
├── 1. Data
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
│   ├── DreamBooth-LoRA
│   └── LoRA
└── README.md
```
