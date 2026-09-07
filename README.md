# Brain Computer Interface for Autonomous Navigation 

![Vue](https://img.shields.io/badge/Vue.js-35495E?logo=vuedotjs&logoColor=4FC08D)
![Vite](https://img.shields.io/badge/Vite-646CFF?logo=vite&logoColor=white)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-AGPL--3.0-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![MNE-Python](https://img.shields.io/badge/MNE--Python-blue)

This GitHub repository constitutes the codebase for my thesis, and final year bachelor's project, at the University of Edinburgh. This was completed in partial fulfilment of my degree from September 2025 to April 2026. My supervisor as a part of this project was *Prof. Dr. Tughrul Arslan*. 

## Publication 

**The people behind this project have decided to publish their findings and are in the process of doing so. Please note that this repository will be updated once this is complete, with the bib entry for the code in this project.** That said, in the meantime if you find this helpful to your own research or engineering work, please consider giving us a star ⭐ on the repo!

---

## Project Description 

This section is basically brief introduction to what this project is trying to accomplish and the basic prerequisites for this project. 

### Brain Computer Interfaces

These are a subset of *Human Computer Interaction* interfaces, that enable interaction with computing devices and actuators with brain signals. The brain signals are usually *electroencephalography* (EEG) signals collected using dry electrodes, in this study we use the [*PiEEG-16*](https://www.pieeg.com/hardware/pieeg-16) - this link is working as of May 2026.

### This project 

This project uses *Motor Imagery* (MI), to control the interface where the user is expected to imagine some sort of motion pertaining to a specific intent. The interface classifies this intent and sends said classification to the actuator to take action accordingly. **Our specific task is to enable spatial navigation using BCIs**. This is a standard task and can be applied in many ways, therefore there are varied actuators that can make use of this classification task. This project aims to write general software for this task and demonstrate it on a web-based software interface.

### Relevant Methodology

This project follows a standard BCI pipeline that can broadly be broken down into three steps: (1) Data Acquisition and Transformation (2) Signal Processing, (3) Machine Learning (Deep Learning) and (4) Evaluation. In a nutshell, the study trains eight seminal models to compare to a novel cross attention based model (see [the following section for more info](#multimodal-models-and-the-cross-modal-model)) and improves on existing BCI navigation system's designs by implicitly including a rest state. The code for those is included in the [`/sever`](/server/) side of this monorepo codebase. 


### Multimodal models and the **Cross Modal Model**.

EEG Signals, like most digitally sampled signals, can be analysed in the frequency domain through various frequency transforms, notably the Welch method of estimating the Fourier Transform. There are also Wavelet Transforms that form a time-frequency representation of the EEG signal, often called the image modality, in total giving us three distinct modalities of EEG data:

1. Time-domain representation or the Time Modality (raw EEG signals).
2. Frequency-domain representation or the Frequency Modality.
3. Time-Frequency-domain representation or the Image Modality.

When these three are analysed simultaneously in three separate branches of a neural network, such an EEG analysis model is referred to as a *multimodal* model. These often naively concatenate the output features from each branch however, which is improved via cross-attention in this project and in this repo this novel model might be referred to as the cross-modal model for short.  


## PiEEG 16 

BCIs are widely considered inaccessible due to expensive EEG equipment<sup>[1](    
https://doi.org/10.48550/arXiv.2409.07491)</sup>. PiEEG-16 is an effort to provide relatively inexpensive at-home EEG devices using the Raspberry Pi. This device is used in this study for the demo, mainly because it is more accessible.

--- 

## How to Run the Interface

> [!NOTE]
>
> Please do note that this is a demonstration repository, the BCI itself can only run with the [PiEEG-16](#pieeg-16) device. For convenience, I've bound <kbd>W</kbd>, <kbd>A</kbd>, <kbd>S</kbd> and <kbd>D</kbd> to simulate the *Up, Down, Left and Right* inputs. Additionally, this project was entirely coded by me and this codebase was maintained by me, so I've really only catered to my needs and not what someone using my app might need. This is because the focus was the cross-attention model and making something novel, not the demonstration interface, as such I can imagine you might face some difficulties getting this interface running. Please do not hesitate to contact me to help set this up for you! See my profile to reach out, I look forward to hearing from anyone that might be interested in talking about this project!

### Deployment 

Update: As I've stated, while you are not connected to the PiEEG-16 device, which will be the case for most people, you are unable to run the interface as designed. To enable for you to use it, the interface is redesigned to work with sample inputs (see [here note above for more info](#how-to-run-the-interface)). This bit has been [deployed on Vercel](https://brain-computer-interface-project-epxy3mxm8.vercel.app/) for convenient use. Please note this has been done as of 2026, I will probably not maintain this if you're looking at this years after the fact.


> [!TIP]
>
>Prerequisites: Install [`node.js`](https://nodejs.org/en). You will also need Python, I recommend using `uv` for this which you can [read about and install from here](https://docs.astral.sh/uv/). You need to have `git` installed. Follow the instructions [from the official `git` website](https://git-scm.com/install/).  

The repository follows a standard monorepo project structure, the frontend is made using VueJS and that is contained by the client directory and the backend contains all the ML models and the communication-to-the-frontend driver code.

Run these commands to start the frontend: 

```sh
git clone <repository-url>
cd brain-computer-interface-project/client 
npm run dev 
```

Run these commands to start the backend:

```sh
# If you are executing this in order then cd out to the parent directory
cd .. 

cd server 
uv run fastapi run main.py
```

> [!CAUTION]
>
> Disclaimer: To run it on the PiEEG device, it required access to your network for webhook communication, as this was mainly a demo and a POC. Please reach out to me directly via my profile or the corresponding author on the [publication](#publication) for assistance with setting this up.

## The PyTorch Modules 

The `/server` directory hosts all the PyTorch definitions. While I've tried my best to adhere to standard programming practice, this was always meant to be a personal repository so documentation can only be found where I was concerned I might forget certain decisions I've made in the future. The structure itself is pretty simple:

- All data processing and interfacing primitives are in the [`/server/app/data`](/server/app/data/) directory.
    - Contains four different modules, they're all explained in their respective top comment and are fairly self explanatory. 
- All models are in the [`/server/app/deep-learning-models`](/server/app/deep-learning-models/) directory.
    - Inside this there are three files: `common.py`, `models.py` and `multimodal_models.py`. Please see the top comment in all of them to know what they contain. The least self-explanatory of the bunch is `common.py` so I will explain that briefly; this module contains the common training primitives and utilities such as an early stopping module and a common training interface. 

These are simply definitions, actual driver code was written in Kaggle, saved on Kaggle, worked on with Kaggle and most of the results are on Kaggle. 

### A Note on AI Coding 

> [!IMPORTANT]
>
> AI was not used to author this repository. It was used in a minor capacity to review code, do sanity checks, help with documentation and clean stuff up when needed. The following statement elaborates on the AI Policy of this repository.

I hope you notice there is no `AGENTS.md` or `CLAUDE.md` or whatever the latest agentic AI context provision nominal file-name is. This is for two equally important reasons: (1) This is a graded university assignment, I would not use Agentic AI Coding tools to write everything. (2) AI is *not* good for the environment, current tools do not sufficiently show that their data is ethically sourced and it actively degrades your skill as a developer. I am not completely opposed to AI and use it as a utility when necessary, and it has been used somewhat in this repository too. That said, I do not like offloading my intellectual capacity to it entirely. The purpose of including this statement is mainly that since this repository is now open sourced under the AGPL 3.0 license, there is a non-zero chance someone might want to make a contribution. If that is you, please respect this stance and avoid using Codex, Claude Code and `AGENT.md` styles of agentic coding. Thank you!

---

## References 

```bib
@article{rakhmatulin_pieeg-16_2024,
    title = {{PiEEG}-16 to {Measure} 16 {EEG} {Channels} with {Raspberry} {Pi} for {Brain}-{Computer} {Interfaces} and {EEG} devices},
    url = {http://arxiv.org/abs/2409.07491},
    doi = {10.48550/arXiv.2409.07491},
    urldate = {2026-03-13},
    publisher = {arXiv},
    author = {Rakhmatulin, Ildar},
    month = sep,
    year = {2024},
    note = {arXiv:2409.07491 [eess]},
    keywords = {Electrical Engineering and Systems Science - Signal Processing, Quantitative Biology - Neurons and Cognition},
}
```