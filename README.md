# Proxy-based Masking Module for Revealing Relevance of Characteristics in Motor Imagery

This repository is the official implementations of PMM in pytorch-lightning style:

## Architecture
![image](https://github.com/hyeonthan/pmm/assets/74582262/d1bb916f-332a-47c4-b485-1d87a085a036)

## Abstract
> Brain-computer interface (BCI) has been developed for communication between users and external devices by reflecting users' status and intentions. Motor imagery (MI) is one of the BCI paradigms for controlling external devices by imagining muscle movements. MI-based EEG signals generally tend to contain signals with sparse MI characteristics (sparse MI signals). When conducting domain adaptation (DA) on MI signals with sparse MI signals, it could interrupt the training process. In this paper, we proposed the proxy-based masking module (PMM) for masking sparse MI signals within MI signals. The proposed module was designed to suppress the amplitude of sparse MI signals using the negative similarity-based mask generated between the proxy of rest signals and the feature vectors of MI signals. We attached our proposed module to the conventional DA methods (i.e., the DJDAN, the MAAN, and the DRDA) to verify the effectiveness in the cross-subject environment on dataset 2a of BCI competition IV. When our proposed module was attached to each conventional DA method, the average accuracy was improved by much as 4.67 %, 0.76 %, and 1.72 %, respectively. Hence, we demonstrated that our proposed module could emphasize the information related to MI characteristics.

## 1. Installation

### 1.1 Clone this repository

```bash
$ git clone https://github.com/hyeonthan/pmm.git
```

### 1.2 Environment setup

```bash
$ cd pmm/Dockerfile
$ make start
$ docker exec -it pmm bash
$ cd pmm
```

### 1.3 Preparing data

> After downloading the [BCI Competition IV 2a](https://www.bbci.de/competition/iv/#download) data, make the data's directory in root/datasets/BCI_Competition_IV/2a and root/datasets/preprocess_data_250/2a_6sd_front

```python
BASE_PATH = "/opt/pmm/datasets/preprocess_data_250/2a_6sd_front"
```

## 2. Performance

> Comparison of the performances with the conventional DA methods including the DJDAN, the MAAN, and the DRDA.


![image](https://github.com/hyeonthan/pmm/assets/74582262/77c1b847-8ee9-4e98-aad3-f300a6e1b1e0)


## 3. Feature visualization

> Feature visualization using the _t_–SNE for S2 and S3 on dataset 2a. <br>(a): w/o (left) and w/ (right) the PMM for S2. (b): w/o (left) and w/ (right) the PMM for S3. Blue, orange, green, and red colors indicate “L”, “R”, “F”, and “T”, respectively.

<img src = "https://github.com/hyeonthan/pmm/assets/74582262/c0bf4511-aa21-4928-b0dc-ddd5463f4da4" width="70%" height="70%">




