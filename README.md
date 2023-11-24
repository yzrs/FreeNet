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
  - You can find this in "./info/keypoints_definition.json"
- Purpose:
  - I hope to find that on this mix dataset, the model doesn't perform well on those rare keypoints except for some unusual semantic keypoints like neck.
  - I also hope to find that on this mix dataset, the model performs better than those on the original small dataset.

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
  - tigdog_horse
    - annotations
    - data
      - horse
        - 1.jpg
        - 2.jpg
        - ...
  - tigdog_tiger
    - annotations
    - data
      - tiger
        - 1.jpg
        - 2.jpg
        - ...
- HRNet
  - SL_mix_keypoints.py
  - ...
  - ...


#### Procedure
1. initialization
2. load pretrained weight for HRNet-w32
3. convert label format of different datasets to a unified format of Mix Dataset
4. do transformation on images and targets.
    - half body
    - scale rotation resize
    - color jitter
    - normalization
5. Get the imgs and gts by dataloader, feed it to the network and compute its MSE
6. Scaler step
7. For evaluation:
    - do prediction on the original img -> logit_ori
    - flip the img 
    - do prediction on the flipped img -> logit_flip
    - flip the logit_flip
    - get the final preds by the weighted sum of logit_ori and logit_flip

#### For Mix Dataset
- AP-10K [train], Animal-Pose [train], TigDog-horse [train], TigDog-tiger [train]
- AP-10K [val], AP-10K [test], Animal-Pose [val], TigDog-horse [val], TigDog-tiger [val]
- For training
  - First figure out where the current img is from by the ['dataset'] and ['mode'] filed of the label.
  - Convert the info format to a unified format for mix dataset with 26 semantic keypoints, depends on the info from last step
  - do prediction and compute its loss -> scaler step
- For validating (Evaluate on each dataset one by one)
  - First figure out how many and what datasets do we have in this Mix Dataset
  - Regard one subset as one independent subset and do prediction on this dataset.Then save the preds results
  - Load the preds results info and compute its corresponding performance values (OKS and PCK@0.05) 
  - Do the same thing on other datasets and save the performance value into one list
  - Compute the weighted sum of OKS and PCK@0.05 performance on all datasets and get the final OKS and PCK@0.05


### Label Free
- We want to get a small but more dense dataset by mixing some datasets with different keypoints definition.
- Our model trained on this dataset can achieve a better performance on common keypoints than most datasets like AP-10K,Animal-Pose,tigdog.
- Our model trained on this dataset can achieve a comparable performance on special keypoints than the model trained on the specific dataset.
- Our model can get some accurate results so that we will get more and more dataset by a free way.

#### Why Free?
- Annotations driven based on new requirements is expensive and time-consuming.

#### Challenge?
- Images from different datasets have different styles and resolution, which will hurt the performance of our model
- Animal numbers from different datasets are not balanced, which will make out model do better on specific animals but worse on other animals
- Different datasets have different definition for one specific semantic keypoints.
  - Paws:
    - In AP-10K, the paw keypoints are located in the center of the paws.
    - However, in Tigdog, the paw keypoints are located in the top of the paws.
  - Neck:
    - The definition of neck in Tigdog is far away from the definition in the AP-10K
- Some semantic keypoints are difficult to annotate them accurately
  - Shoulders
  - Hips
  - Neck
  - Wither
  - Throat
- 并不是说难标的语义点数量就一定会少，pck更低
  - 比如AP-10K中的neck虽然难以标记，但数量很多
  - 数量少并不意味着pck低
  - 数量不仅和标记难度相关，还和动物的姿态相关
  - 需要聚焦于数量少且有一定pck提升空间的点
  - 易学 -> 提升空间较小
  - 学习难度中等 -> 需着重考虑
  - 难学 + 数量少 -> 提升空间较大，需着重考虑
  - 难学 + 数量多 -> 提升空间较小
  
#### Idea?
- 每个点都采样3K，用每个点的NME的结果指代其学习难度
  - 左: 易学 + 学习难度中等/数量多
  - 中: 学习难度中等/数量少 + 难学/数量多
  - 右: 难学/数量少
- Few-shot 
  - 选取图片时，尽量选择含有大量学习难度中等的点的图片，易学和难学的点的数量应控制在较低水平
  - 选取图片时，还应考虑动物数量平衡的关系
  - 混合时，还应考虑每种数据集之间数量均衡的关系

#### Stage Results
- 点之间是相互独立的，特征相似和位置相邻并不会产生影响
- 动物种类数量的不平衡会造成少量影响
- 混合数据集语义点定义的不一致可能会造成少量性能的下降，但是会被数量带来的提升所弥补
- 