# AdaptiveStep: Automatically Dividing Reasoning Step through Model Confidence

  - [2025/1/31]  We have published our paper here.
  
  - [2025/1/31]  We have released our model and data [here](https://huggingface.co/Lux0926).
  
## Environment Setup

### Train enviroment
  Our PRM training scripts based on version 0.4.5 of openrlhf. If you want to use our training scripts, please run the following code to create the conda environment.
  ```bash
  conda create -n asprm_train python=3.10.0
  conda activate asprm_train
  pip install openrlhf==0.4.5
  ```

### Eval enviroment

## Trarning Code
In the `train` folder, we have provided the scripts used for training PRM. To replicate our training process, please run the scripts in the `train` directory after setting up the training environment.

#### Example

  ```bash
  cd train/
  bash train_ASPRM-M.sh
  ```

## Evaluation
