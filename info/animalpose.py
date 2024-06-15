dataset_info = dict(
    dataset_name='animalpose',
    paper_info=dict(
        author='Cao, Jinkun and Tang, Hongyang and Fang, Hao-Shu and '
        'Shen, Xiaoyong and Lu, Cewu and Tai, Yu-Wing',
        title='Cross-Domain Adaptation for Animal Pose Estimation',
        container='The IEEE International Conference on '
        'Computer Vision (ICCV)',
        year='2019'
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
        dict(name='Nose', id=2, color=[51, 153, 255], type='upper', swap=''),
        3:
        dict(
            name='L_ear',
            id=3,
            color=[0, 255, 0],
            type='upper',
            swap='R_ear'),
        4:
        dict(
            name='R_ear',
            id=4,
            color=[0, 255, 0],
            type='upper',
            swap='L_ear'),
        5:
        dict(
            name='L_F_hip',
            id=5,
            color=[51, 153, 255],
            type='upper',
            swap='R_F_hip'),
        6:
        dict(
            name='R_F_hip',
            id=6,
            color=[0, 255, 0],
            type='upper',
            swap='L_F_hip'),
        7:
        dict(
            name='L_B_hip',
            id=7,
            color=[255, 128, 0],
            type='lower',
            swap='R_B_hip'),
        8:
        dict(
            name='R_B_hip', id=8, color=[0, 255, 0], type='lower',
            swap='L_B_hip'),
        9:
        dict(
            name='L_F_knee',
            id=9,
            color=[51, 153, 255],
            type='upper',
            swap='R_F_knee'),
        10:
        dict(
            name='R_F_knee',
            id=10,
            color=[255, 128, 0],
            type='upper',
            swap='L_F_knee'),
        11:
        dict(
            name='L_B_knee',
            id=11,
            color=[255, 128, 0],
            type='lower',
            swap='R_B_knee'),
        12:
        dict(
            name='R_B_knee',
            id=12,
            color=[0, 255, 0],
            type='lower',
            swap='L_B_knee'),
        13:
        dict(
            name='L_F_paw',
            id=13,
            color=[0, 255, 0],
            type='upper',
            swap='R_F_paw'),
        14:
        dict(
            name='R_F_paw',
            id=14,
            color=[0, 255, 0],
            type='lower',
            swap='L_F_paw'),
        15:
        dict(
            name='L_B_paw',
            id=15,
            color=[0, 255, 0],
            type='lower',
            swap='R_B_paw'),
        16:
        dict(
            name='R_B_paw',
            id=16,
            color=[0, 255, 0],
            type='lower',
            swap='L_B_paw'),
        17:
        dict(
            name='Throat',
            id=17,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        18:
        dict(
            name='Wither',
            id=18,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        19:
        dict(
            name='Tail',
            id=19,
            color=[51, 153, 255],
            type='lower',
            swap=''),
    },
    skeleton_info={
        0: dict(link=('L_eye', 'R_eye'), id=0, color=[0, 0, 255]),
        1: dict(link=('L_eye', 'Nose'), id=1, color=[0, 0, 255]),
        2: dict(link=('R_eye', 'Nose'), id=2, color=[0, 0, 255]),
        3: dict(link=('Nose', 'Throat'), id=3, color=[0, 0, 255]),
        4: dict(link=('Throat', 'Wither'), id=4, color=[0, 0, 255]),
        5: dict(link=('Wither', 'L_F_hip'), id=5, color=[0, 0, 255]),
        6: dict(link=('Wither', 'R_F_hip'), id=6, color=[0, 0, 255]),
        7: dict(link=('L_F_hip', 'L_F_knee'), id=7, color=[0, 255, 255]),
        8: dict(link=('L_F_knee', 'L_F_paw'), id=8, color=[0, 255, 255]),
        9: dict(link=('R_F_hip', 'R_F_knee'), id=9, color=[6, 156, 250]),
        10: dict(link=('R_F_knee', 'R_F_paw'), id=10, color=[6, 156, 250]),
        12: dict(link=('Wither', 'Tail'), id=11, color=[0, 255, 255]),
        13: dict(link=('Tail', 'L_B_hip'), id=12, color=[0, 255, 255]),
        14: dict(link=('L_B_hip', 'L_B_knee'), id=13, color=[0, 255, 255]),
        15: dict(link=('L_B_knee', 'L_B_paw'), id=14, color=[0, 255, 255]),
        16: dict(link=('Tail', 'R_B_hip'), id=15, color=[6, 156, 250]),
        17: dict(link=('R_B_hip', 'R_B_knee'), id=16, color=[6, 156, 250]),
        18: dict(link=('R_B_knee', 'R_B_paw'), id=17, color=[6, 156, 250]),
        19: dict(link=('L_eye', 'L_ear'), id=18, color=[0, 0, 255]),
        20: dict(link=('R_eye', 'R_ear'), id=19, color=[0, 0, 255]),
    },
    joint_weights=[
        1., 1., 1., 1., 1.,
        1., 1., 1., 1.,
        1.2, 1.2, 1.2, 1.2,
        1.5, 1.5, 1.5, 1.5,
        1., 1., 1.
    ],

    # Note: The original paper did not provide enough information about
    sigmas=[
        0.025, 0.025, 0.026, 0.035, 0.035,
        0.079, 0.079, 0.107, 0.107,
        0.072, 0.072, 0.087, 0.087,
        0.062, 0.062, 0.089, 0.089,
        0.10, 0.10, 0.035
    ]
)
