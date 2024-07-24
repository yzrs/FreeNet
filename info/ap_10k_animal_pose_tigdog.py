dataset_info = dict(
    dataset_name='ap_10k_animal_pose_tigdog',
    paper_info=dict(
        author='Yu, Hang and Xu, Yufei and Zhang, Jing and '
        'Zhao, Wei and Guan, Ziyu and Tao, Dacheng',
        title='AP-10K: A Benchmark for Animal Pose Estimation in the Wild',
        container='35th Conference on Neural Information Processing Systems '
        '(NeurIPS 2021) Track on Datasets and Bench-marks.',
        year='2021',
        homepage='https://github.com/AlexTheBad/AP-10K',
    ),
    keypoint_info={
        0:
        dict(
            name='L_eye', id=0, color=[0, 255, 0], type='upper', swap='R_eye'),
        1:
        dict(
            name='R_eye',
            id=1,
            color=[255, 128, 0],
            type='upper',
            swap='L_eye'),
        2:
        dict(name='nose', id=2, color=[51, 153, 255], type='upper', swap=''),
        3:
        dict(name='neck', id=3, color=[51, 153, 255], type='upper', swap=''),
        4:
        dict(
            name='tail',
            id=4,
            color=[51, 153, 255],
            type='lower',
            swap=''),
        5:
        dict(
            name='L_F_hip',
            id=5,
            color=[51, 153, 255],
            type='upper',
            swap='R_F_hip'),
        6:
        dict(
            name='L_F_knee',
            id=6,
            color=[51, 153, 255],
            type='upper',
            swap='R_F_knee'),
        7:
        dict(
            name='L_F_paw',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='R_F_paw'),
        8:
        dict(
            name='R_F_hip',
            id=8,
            color=[0, 255, 0],
            type='upper',
            swap='L_F_hip'),
        9:
        dict(
            name='R_F_knee',
            id=9,
            color=[255, 128, 0],
            type='upper',
            swap='L_F_knee'),
        10:
        dict(
            name='R_F_paw',
            id=10,
            color=[0, 255, 0],
            type='lower',
            swap='L_F_paw'),
        11:
        dict(
            name='L_B_hip',
            id=11,
            color=[255, 128, 0],
            type='lower',
            swap='R_B_hip'),
        12:
        dict(
            name='L_B_knee',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap='R_B_knee'),
        13:
        dict(
            name='L_B_paw',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap='R_B_paw'),
        14:
        dict(
            name='R_B_hip', id=14, color=[0, 255, 0], type='lower',
            swap='L_B_hip'),
        15:
        dict(
            name='R_B_knee',
            id=15,
            color=[0, 255, 0],
            type='lower',
            swap='L_B_knee'),
        16:
        dict(
            name='R_B_paw',
            id=16,
            color=[0, 255, 0],
            type='lower',
            swap='L_B_paw'),
        17:
        dict(
            name='L_ear',
            id=17,
            color=[0, 255, 0],
            type='upper',
            swap='R_ear'),
        18:
        dict(
            name='R_ear',
            id=18,
            color=[0, 255, 0],
            type='upper',
            swap='L_ear'),
        19:
        dict(
            name='throat',
            id=19,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        20:
        dict(
            name='wither',
            id=20,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        21:
        dict(
            name='chin',
            id=21,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        22:
        dict(
            name='L_shoulder',
            id=22,
            color=[0, 255, 0],
            type='upper',
            swap='R_shoulder'),
        23:
        dict(
            name='R_shoulder',
            id=23,
            color=[0, 255, 0],
            type='upper',
            swap='L_shoulder'),
    },
    skeleton_info={
        0: dict(link=('L_eye', 'R_eye'), id=0, color=[0, 0, 255]),
        1: dict(link=('L_eye', 'nose'), id=1, color=[0, 0, 255]),
        2: dict(link=('R_eye', 'nose'), id=2, color=[0, 0, 255]),
        3: dict(link=('nose', 'neck'), id=3, color=[0, 255, 0]),
        4: dict(link=('neck', 'tail'), id=4, color=[0, 255, 0]),
        5: dict(link=('neck', 'L_F_hip'), id=5, color=[0, 255, 255]),
        6: dict(link=('L_F_hip', 'L_F_knee'), id=6, color=[0, 255, 255]),
        7: dict(link=('L_F_knee', 'L_F_paw'), id=6, color=[0, 255, 255]),
        8: dict(link=('neck', 'R_F_hip'), id=7, color=[6, 156, 250]),
        9: dict(link=('R_F_hip', 'R_F_knee'), id=8, color=[6, 156, 250]),
        10: dict(link=('R_F_knee', 'R_F_paw'), id=9, color=[6, 156, 250]),
        11: dict(link=('tail', 'L_B_hip'), id=10, color=[0, 255, 255]),
        12: dict(link=('L_B_hip', 'L_B_knee'), id=11, color=[0, 255, 255]),
        13: dict(link=('L_B_knee', 'L_B_paw'), id=12, color=[0, 255, 255]),
        14: dict(link=('tail', 'R_B_hip'), id=13, color=[6, 156, 250]),
        15: dict(link=('R_B_hip', 'R_B_knee'), id=14, color=[6, 156, 250]),
        16: dict(link=('R_B_knee', 'R_B_paw'), id=15, color=[6, 156, 250]),
        17: dict(link=('L_eye', 'L_ear'), id=16, color=[0, 0, 255]),
        18: dict(link=('R_eye', 'R_ear'), id=17, color=[0, 0, 255]),
    },
    joint_weights=[
        1., 1., 1., 1., 1.,
        1., 1.2, 1.5,
        1., 1.2, 1.5,
        1., 1.2, 1.5,
        1., 1.2, 1.5,
        1., 1., 1., 1.,
        1., 1., 1.
    ],

    # Note: The original paper did not provide enough information about
    # the sigmas. We modified from 'https://github.com/cocodataset/'
    # 'cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py#L523'
    sigmas=[
        0.025, 0.025, 0.026, 0.035, 0.035,
        0.079, 0.072, 0.062,
        0.079, 0.072,0.062,
        0.107, 0.087, 0.089,
        0.107, 0.087, 0.089,
        0.035, 0.035, 0.10, 0.10,
        0.026, 0.035, 0.035,
    ],
    # from animal pose keypoint definition to union definition
    animal_pose_2_union=[[0,0],[1,1],[2,2],[3,17],[4,18],
                         [5,5],[6,8],[7,11],[8,14],
                         [9,6],[10,9],[11,12],[12,15],
                         [13,7],[14,10],[15,13],[16,16],
                         [17,19],[18,20],[19,4]],
    # from ap_10k keypoint definition to union definition
    ap_10k_2_union=[[0,0],[1,1],[2,2],[3,3],[4,4],
                    [5,5],[6,6],[7,7],[8,8],
                    [9,9],[10,10],[11,11],[12,12],
                    [13,13],[14,14],[15,15],[16,16]],
    tigdog_2_union=[[0,0],[1,1],[2,21],
                    [3,7],[4,10],[5,13],[6,16],
                    [7,4],
                    [8,6],[9,9],[10,12],[11,15],
                    [12,22],[13,23],
                    [14,5],[15,8],[16,11],[17,14],
                    [18,3]]
)
