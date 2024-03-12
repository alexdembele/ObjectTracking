from PIL import Image
from torchvision import models, transforms
from collections import defaultdict
from torch import norm
import numpy as np

model = models.resnet50(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionner l'image à la taille attendue par ResNet
    transforms.ToTensor(),           # Convertir l'image en un tenseur PyTorch
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Normaliser l'image selon les valeurs moyennes et d'écart-type de l'ensemble d'entraînement
        std=[0.229, 0.224, 0.225]
    )
])



def getDeepFeatures(frame):
    image_pil = Image.fromarray(frame)
    input = transform(image_pil).unsqueeze(0)

    out = model.conv1(input)
    out = model.bn1(out)
    out = model.relu(out)
    out = model.maxpool(out)
    out = model.layer1(out)
    return out

# def buildRVector(roi):
#     hei = roi.shape[0]//5
#     wid = roi.shape[1]//5
#     R_vector= defaultdict(list)
#     for i in range(hei):
#         for j in range(wid):
#             tamp = np.zeros((5*hei,5*wid))
#             tamp[i*hei:(i+1)*hei,j*wid:(j+1)*wid]=roi[i*hei:(i+1)*hei,j*wid:(j+1)*wid]
#     R_vector[i*hei+j] = getDeepFeatures(tamp)
#     return R_vector,hei,wid



def buildAccumulator(frame,R_vector,hei,wid):
    frame_extended= np.zeros((frame.shape[0]+2*hei,frame.shape[1]+2*wid,frame.shape[2]))
    frame_extended[hei:hei+frame.shape[0],wid:frame.shape[1]+wid,:] = frame
    acc = np.zeros_like(frame)
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):

            inputs = frame_extended[i+hei//2:i+3*hei//2,j+wid//2:j+3*wid//2,:].astype(np.uint8)

            feature = getDeepFeatures(inputs)
            acc[i,j] = norm((feature-R_vector).detach())
    return acc

def n_min(a, n):
    '''
    Return the N max elements and indices in a
    '''
    indices = a.ravel().argsort()[:n]
    indices = (np.unravel_index(i, a.shape) for i in indices)
    return [(a[i], i) for i in indices]
