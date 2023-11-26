# FreeNet
# HRNet For Animal Pose Estimation
animal pose estimation with HRNet-w32
The datasets won't will be uploaded here.
AP-10K:
    split1
Animal-Pose
    train:val = 9:1 for every category of animal
TigDog
    correct some error landmarks of tiger(shoulders and hips)

#### Mix Dataset Experiment
26 KPS:
- Train Dataset: 
  - AP-10K_train 
  - Animal-Pose_train
  - TigDog_train
- Val Dataset: 
  - AP-10K_val
  - AP-10K_test
  - Animal-Pose_val
  - TigDog_val
- Keypoint Definition
  - You can find this in "info/keypoints_definition.json"
  
21 KPS:
- Train Dataset: 
  - AP-10K_train 
  - Animal-Pose_train
- Val Dataset: 
  - AP-10K_val
  - AP-10K_test
  - Animal-Pose_val
- Keypoint Definition
  - You can find this in "info/ap_10k_animal_pose_union_keypoints_format.json"

#### Dataset Path
- Dataset
  - ap_10k
    - annotations
    - data
      - 1.jpg
      - 2.jpg
      - ...
  - animal_pose
    - annotations
    - data
      - 1.jpg
      - 2.jpg
      - ...
  - tigdog
    - annotations
    - data
      - horse
        - 1.jpg
        - 2.jpg
        - ...
      - tiger
        - 1.jpg
        - 2.jpg
        - ...

- freenet
  - ours.py
  - ...
  - ...

#### Some Folders and Files Description
**Label Files Renaming is Needed.**
**For example, ap_10k_train.json / ap_10k_val.json / ap_10k_test.json**
**For example, animal_pose_train.json / animal_pose_val.json**
**For example, tigdog_train.json / tigdog_val.json**
**For example, ap_10k_animal_pose_union_train.json / ap_10k_animal_pose_union_val.json**
**Some default args params like steps are used for debug. Modify them for running.**
- experiment 
  where we save the files during training
- info
  gt format and labels info
- train_utils
  some utils function used in training
- outer_tools
  Although we have our own PCK evaluating method, we hope to be consistent to other authors.
  So the outer_tools is for this purpose.
- eval_ap10k / eval_ap10k_animalpose
  to eval model on different datasets.Notice that we can only use eval_ap10k.py to test the performance of the model with
  17 prediction heads and eval_ap10k_animalpose to test the performance with 21 prediction heads.
- ours_ap10k_animalpose
  Train model(Pretrained on label data) on 10% AP-10K + Animal Pose Datasets
- ours_split_ap_10k
  Train for share different part of keypoints in AP-10K
  Parameter level is required here.
  Level = 0 -> share eyes,nose,neck,tail
  Level = 1 -> share Level 0 + paws
  Level = 2 -> share Level 1 + hips
- ours_synthetic_ap_10k
  Train for modify 25 imgs per animal setting to be consistent with the 5 imgs per animal setting in ScarceNet.The only 
  exception is the neck and tail(Remain)
- finetune
  finetune the best model 
- Pretrained Weights
  - 25_5_imgs_SL_hrnet_pretrained.pth : for 25 imgs per animal setting which is consistent with the 5 imgs per animal setting
  - ap_10k_animal_pose_mix_SL_0.1_hrnet.pth : for AP-10K + Animal Pose 10% setting
  - ap_10k_split_level_0/1/2.pth : for the setting of different levels of splitting ap_10k 
  - pretrained_ori.pth : It is same as the hrnet_w32-36af842e.pth (Pretrained on ImageNet)
  - scarcenet.pth : pretrained_weights for our finetune
- Saved Weights
  - ours.pth : ours for 25 imgs per animal setting which is consistent with the 5 imgs per animal setting
  - ours_finetune.pth : ours finetune results for 25 imgs per animal setting which is consistent with the 5 imgs per animal setting
  - ls / ls_lu / ls_lu_lf : Training Results for AP-10K + Animal Pose 10%
  - ours_level_0/1/2 : Ours Results for different levels of splitting ap_10k 

#### sh
total-steps could be modified appropriately.

python ours_ap_10k_animal_pose.py \
    --workers 8 \
    --seed 2 \
    --batch-size 8 \
    --mu 7 \
    --total-steps 45000 \
    --eval-step 150 \
    --warmup-steps 900 \
    --down-step 9000 \
    --feedback-steps-start 4500 \
    --feedback-steps-complete 6000 \
    --amp

python ours_synthetic_ap_10k.py \
    --workers 8 \
    --seed 2 \
    --batch-size 8 \
    --mu 7 \
    --total-steps 45000 \
    --eval-step 150 \
    --warmup-steps 900 \
    --down-step 9000 \
    --feedback-steps-start 4500 \
    --feedback-steps-complete 6000 \
    --amp

level -> 0 / 1 / 2
python ours_split_ap_10k.py \
    --level 0 \ 
    --workers 8 \
    --seed 2 \
    --batch-size 32 \
    --mu 1 \
    --total-steps 45000 \
    --eval-step 150 \
    --warmup-steps 900 \
    --down-step 9000 \
    --feedback-steps-start 4500 \
    --feedback-steps-complete 6000 \
    --amp