import torch

checkpoint_path = 'model.pth'
ckpt = torch.load(checkpoint_path)
model_weights = ckpt['state_dict']
# model_weights = ckpt['student_model']
# model_weights = ckpt['teacher_model']
torch.save(model_weights, 'model.pth')
print('done')