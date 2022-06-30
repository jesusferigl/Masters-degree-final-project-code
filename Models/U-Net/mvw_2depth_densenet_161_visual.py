#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pickle
from sklearn.model_selection import train_test_split
import time
import glob
import PIL
from PIL import Image
import copy
import cv2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from comet_ml import Experiment


# In[3]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[4]:


hiperparametros = {
    'epocas' : 1000,
    'batch_size' : 32,
    'mascaras_train' : glob.glob('mascaras_train/*'),
    'learning_rate' : 0.001,
    'weight_decay' : 1e-5,
    'weights' : [0.075, 0.925],
    'arquitectura' : 'densenet161',
}


# In[5]:


experiment = Experiment(
    api_key="RrpYLF8eASZofs2sNQ8hDHgnL",
    project_name="u-nets-paper-msc",
    workspace="jesusfernandeziglesias",
)
experiment.set_name('MVW-2Depth-DenseNet-161-Visual')
experiment.log_parameters(hiperparametros)


# In[6]:


class MultiviewNet(torch.nn.Module):
    def __init__(self, name_second_network):
        super().__init__()
        self.conv_1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 6, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU()
        )
        self.conv_2_1 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 9, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(9),
            torch.nn.ReLU()
        )
        self.conv_1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 6, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU()
        )
        self.conv_2_2 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 9, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(9),
            torch.nn.ReLU()
        )
        self.conv_1_3 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 6, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU()
        )
        self.conv_2_3 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 9, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(9),
            torch.nn.ReLU()
        )
        self.conv_1_4 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 6, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU()
        )
        self.conv_2_4 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 9, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(9),
            torch.nn.ReLU()
        )
        self.modelo = smp.Unet(encoder_name = 'densenet161', encoder_weights = 'imagenet', in_channels=36, classes=2)
    def forward(self, x1, x2, x3, x4):
        output_tras_conv_1_1 = self.conv_1_1(x1)
        output_tras_conv_2_1 = self.conv_2_1(output_tras_conv_1_1)
        output_tras_conv_1_2 = self.conv_1_2(x2)
        output_tras_conv_2_2 = self.conv_2_2(output_tras_conv_1_2)
        output_tras_conv_1_3 = self.conv_1_3(x3)
        output_tras_conv_2_3 = self.conv_2_3(output_tras_conv_1_3)
        output_tras_conv_1_4 = self.conv_1_4(x4)
        output_tras_conv_2_4 = self.conv_2_4(output_tras_conv_1_4)
        output_tras_red_1 = torch.cat([output_tras_conv_2_1, output_tras_conv_2_2, output_tras_conv_2_3, output_tras_conv_2_4], axis = 1)
        output = self.modelo(output_tras_red_1)
        return output


# In[7]:


class MyDataset(Dataset):
    def __init__(self, ruta_imagenes_visual, ruta_mascaras):
        self.ruta_imagenes_visual = ruta_imagenes_visual
        self.ruta_mascaras = ruta_mascaras
        
    def __getitem__(self, index):
        resize = torchvision.transforms.Resize((192, 192))
        x_HJI = np.array((PIL.Image.open(self.ruta_imagenes_visual + self.ruta_mascaras[index].split('/')[-1][:-6] + '.HJI.png')))
        x_HJV = np.array((PIL.Image.open(self.ruta_imagenes_visual + self.ruta_mascaras[index].split('/')[-1][:-6] + '.HJV.png')))
        x_HIV = np.array((PIL.Image.open(self.ruta_imagenes_visual + self.ruta_mascaras[index].split('/')[-1][:-6] + '.HIV.png')))
        x_JIV = np.array((PIL.Image.open(self.ruta_imagenes_visual + self.ruta_mascaras[index].split('/')[-1][:-6] + '.JIV.png')))
        y = np.array((PIL.Image.open(self.ruta_mascaras[index])))
        x_HJI = np.array(resize(Image.fromarray(x_HJI)))/255
        x_HJV = np.array(resize(Image.fromarray(x_HJV)))/255
        x_HIV = np.array(resize(Image.fromarray(x_HIV)))/255
        x_JIV = np.array(resize(Image.fromarray(x_JIV)))/255
        y = np.array(resize(Image.fromarray(y)))
        x_HJI = np.moveaxis(x_HJI, -1, 0)
        x_HJV = np.moveaxis(x_HJV, -1, 0)
        x_HIV = np.moveaxis(x_HIV, -1, 0)
        x_JIV = np.moveaxis(x_JIV, -1, 0)
        y = torch.from_numpy(y).long()
        x_HJI = torch.from_numpy(x_HJI).float()
        x_HJV = torch.from_numpy(x_HJV).float()
        x_HIV = torch.from_numpy(x_HIV).float()
        x_JIV = torch.from_numpy(x_JIV).float()
        return x_HJI, x_HJV, x_HIV, x_JIV, y
    
    def __len__(self):
        return len(self.ruta_mascaras)
        
class MyDatasetNormalRotationAndFlip(Dataset):
    def __init__(self, ruta_imagenes_visual, ruta_mascaras):
        self.ruta_imagenes_visual = ruta_imagenes_visual
        self.ruta_mascaras = ruta_mascaras
        
    def __getitem__(self, index):
        resize = torchvision.transforms.Resize((192, 192))
        angulo = np.random.choice([0, 90, 180, 270])
        flip = np.random.choice(['0', 'v', 'h'])
        x_HJI = np.array((PIL.Image.open(self.ruta_imagenes_visual + self.ruta_mascaras[index].split('/')[-1][:-6] + '.HJI.png')))
        x_HJV = np.array((PIL.Image.open(self.ruta_imagenes_visual + self.ruta_mascaras[index].split('/')[-1][:-6] + '.HJV.png')))
        x_HIV = np.array((PIL.Image.open(self.ruta_imagenes_visual + self.ruta_mascaras[index].split('/')[-1][:-6] + '.HIV.png')))
        x_JIV = np.array((PIL.Image.open(self.ruta_imagenes_visual + self.ruta_mascaras[index].split('/')[-1][:-6] + '.JIV.png')))
        y = np.array((PIL.Image.open(self.ruta_mascaras[index])))
        x_HJI = np.array(resize(Image.fromarray(x_HJI)))/255
        x_HJV = np.array(resize(Image.fromarray(x_HJV)))/255
        x_HIV = np.array(resize(Image.fromarray(x_HIV)))/255
        x_JIV = np.array(resize(Image.fromarray(x_JIV)))/255
        y = np.array(resize(Image.fromarray(y)))
        if angulo == 0:
            pass
        elif angulo == 90:
            x_HJI = cv2.rotate(x_HJI, cv2.cv2.ROTATE_90_CLOCKWISE)
            x_HJV = cv2.rotate(x_HJV, cv2.cv2.ROTATE_90_CLOCKWISE)
            x_HIV = cv2.rotate(x_HIV, cv2.cv2.ROTATE_90_CLOCKWISE)
            x_JIV = cv2.rotate(x_JIV, cv2.cv2.ROTATE_90_CLOCKWISE)
            y = cv2.rotate(y, cv2.cv2.ROTATE_90_CLOCKWISE)
        elif angulo == 180:
            x_HJI = cv2.rotate(x_HJI, cv2.cv2.ROTATE_180)
            x_HJV = cv2.rotate(x_HJV, cv2.cv2.ROTATE_180)
            x_HIV = cv2.rotate(x_HIV, cv2.cv2.ROTATE_180)
            x_JIV = cv2.rotate(x_JIV, cv2.cv2.ROTATE_180)
            y = cv2.rotate(y, cv2.cv2.ROTATE_180)
        elif angulo == 270:
            x_HJI = cv2.rotate(x_HJI, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_HJV = cv2.rotate(x_HJV, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_HIV = cv2.rotate(x_HIV, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_JIV = cv2.rotate(x_JIV, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            y = cv2.rotate(y, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        if flip == '0':
            pass
        elif flip == 'v':
            x_HJI = cv2.flip(x_HJI, 0)
            x_HJV = cv2.flip(x_HJV, 0)
            x_HIV = cv2.flip(x_HIV, 0)
            x_JIV = cv2.flip(x_JIV, 0)
            y = cv2.flip(y, 0)
        elif flip == 'h':
            x_HJI = cv2.flip(x_HJI, 1)
            x_HJV = cv2.flip(x_HJV, 1)
            x_HIV = cv2.flip(x_HIV, 1)
            x_JIV = cv2.flip(x_JIV, 1)
            y = cv2.flip(y, 1)
        x_HJI = np.moveaxis(x_HJI, -1, 0)
        x_HJV = np.moveaxis(x_HJV, -1, 0)
        x_HIV = np.moveaxis(x_HIV, -1, 0)
        x_JIV = np.moveaxis(x_JIV, -1, 0)
        y = torch.from_numpy(y).long()
        noise_x_HJI = torch.from_numpy(abs(np.random.normal(loc=0, scale=np.random.choice([0.01, 0.05, 0.1]), size=x_HJI.shape))).float()
        noise_x_HJV = torch.from_numpy(abs(np.random.normal(loc=0, scale=np.random.choice([0.01, 0.05, 0.1]), size=x_HJV.shape))).float()
        noise_x_HIV = torch.from_numpy(abs(np.random.normal(loc=0, scale=np.random.choice([0.01, 0.05, 0.1]), size=x_HIV.shape))).float()
        noise_x_JIV = torch.from_numpy(abs(np.random.normal(loc=0, scale=np.random.choice([0.01, 0.05, 0.1]), size=x_JIV.shape))).float()
        x_HJI = torch.from_numpy(x_HJI).float() + noise_x_HJI
        x_HJV = torch.from_numpy(x_HJV).float() + noise_x_HJV
        x_HIV = torch.from_numpy(x_HIV).float() + noise_x_HIV
        x_JIV = torch.from_numpy(x_JIV).float() + noise_x_JIV
        return x_HJI, x_HJV, x_HIV, x_JIV, y
    
    def __len__(self):
        return len(self.ruta_mascaras)


# In[11]:


mascaras_train, mascaras_validacion = train_test_split(hiperparametros['mascaras_train'], test_size=0.2, random_state=25)
train = MyDatasetNormalRotationAndFlip('Imagenes_con_NaN/HIJV/', mascaras_train)
valid = MyDataset('Imagenes_con_NaN/HIJV/', mascaras_validacion)
trainloader = DataLoader(train, batch_size=hiperparametros['batch_size'], shuffle=True)
validloader = DataLoader(valid, batch_size=hiperparametros['batch_size'], shuffle=True)


# In[12]:


print('Tamaño del dataloader de entrenamiento:', trainloader.dataset.__len__())
print('Tamaño del dataloader de validacion:', validloader.dataset.__len__())


# In[13]:


red = MultiviewNet(hiperparametros['arquitectura']).to(device)
class_weights = torch.FloatTensor(hiperparametros['weights']).cuda()
criterion = nn.CrossEntropyLoss(weight = class_weights)
optimizer = torch.optim.Adam(red.parameters(), lr = hiperparametros['learning_rate'], weight_decay = hiperparametros['weight_decay'])


# a, b, c, d, e = valid.__getitem__(10)
# plt.imshow(np.moveaxis(a.cpu().numpy(), 0, -1))
# plt.show()
# plt.imshow(np.moveaxis(b.cpu().numpy(), 0, -1))
# plt.show()
# plt.imshow(np.moveaxis(c.cpu().numpy(), 0, -1))
# plt.show()
# plt.imshow(np.moveaxis(d.cpu().numpy(), 0, -1))
# plt.show()
# plt.imshow(e.cpu().numpy())
# plt.show()

# In[10]:


def accuracy(predb, yb):
    metrica = 0
    for i in range(yb.shape[0]):
        metrica += (predb[i,:,:,:].argmax(dim=0) == yb[i,:,:]).float().mean().item()
    return(metrica/yb.shape[0])

def test_metrics(red):
    red.eval()
    loss_medio_test = 0
    accuracy_medio_test = 0
    pasos_test = 0
    tpl = []
    fnl = []
    fpl = []
    with torch.no_grad():
        for data in validloader:
            inputs1, inputs2, inputs3, inputs4, labels = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)
            outputs = red(inputs1, inputs2, inputs3, inputs4)
            loss_validation = criterion(outputs, labels)
            loss_medio_test += loss_validation.item()
            pasos_test += 1
            accuracy_medio_test += accuracy(outputs, labels)
            for m in range(labels.shape[0]):
                true_positives = 0; false_positives = 0; false_negatives = 0
                mascara = labels[m,:,:]
                output = outputs[m,:,:,:].argmax(dim=0)
                tpl.append(torch.sum(output[mascara == 1] == 1).item())
                fnl.append(torch.sum(output[mascara == 1] == 0).item())
                fpl.append(torch.sum(output[mascara == 0] == 1).item())
    precission = np.mean(np.array(tpl)/(np.array(tpl) + np.array(fpl)))
    recall = np.mean(np.array(tpl)/(np.array(tpl) + np.array(fnl)))
    dice = np.mean(2*np.array(tpl)/(2*np.array(tpl)+np.array(fpl)+np.array(fnl)))
    f = 2*precission*recall/(precission+recall)
    experiment.log_metric('Loss validation:', loss_medio_test/pasos_test)
    experiment.log_metric('Accuracy validation:', accuracy_medio_test/pasos_test)
    experiment.log_metric('Precission validation:', precission)
    experiment.log_metric('Recall validation:', recall)
    experiment.log_metric('Dice validation:', dice)
    experiment.log_metric('F-score validation:', f)
    return dice


def entrena(red):
    tiempo = time.time()
    mejor_modelo = None
    epoca = None
    dice_for_check = 0
    for epoch in range(hiperparametros['epocas']):
        experiment.set_epoch(epoch)
        loss_medio_train = 0
        accuracy_medio_train = 0
        pasos_train = 0
        for i, data in enumerate(trainloader, 0):
            inputs1, inputs2, inputs3, inputs4, labels = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)
            optimizer.zero_grad()
            outputs = red(inputs1, inputs2, inputs3, inputs4)
            loss = criterion(outputs, labels)
            loss_medio_train += loss.item()
            pasos_train += 1
            accuracy_medio_train += accuracy(outputs, labels)
            loss.backward()
            optimizer.step()
        experiment.log_metric('Loss train:', loss_medio_train/pasos_train)
        experiment.log_metric('Accuracy train:', accuracy_medio_train/pasos_train)
        dice = test_metrics(red)
        if dice > dice_for_check:
            mejor_modelo = copy.deepcopy(red)
            epoca = epoch
            dice_for_check = dice
        red.train()
        print('Epoca ', epoch+1, 'completada.')
    print('Finished Training')
    print('Time required:', time.time()-tiempo)
    print('Mejor modelo alcanzado en la epoca ', epoca, ' con un dice en test de ', dice_for_check)
    torch.save(mejor_modelo, 'mejor_modelo_MVW-2Depth-DenseNet-161-Visual')
    torch.save(red, 'ultimo_modelo_MVW-2Depth-DenseNet-161-Visual')


# In[11]:


entrena(red)


# In[ ]:


#experiment.log_notebook('MVW-2Depth-Eff-b6-Visual.py')
experiment.end()

