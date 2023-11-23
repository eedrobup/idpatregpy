# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 21:11:12 2023

For preprocessing raw mobile images data of frogs

@author: Eedrobup
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc,ndimage
from skimage import io,color
from skimage.util import crop

#import config

#working_directory = config.PersonalConfig.call_wd()
os.chdir("C:\\Users\\Eedrobup\\Downloads\\Frog photos")
#%% Import point part

path = 'Label\\C\\'
y1 = np.zeros((len(os.listdir(path)[:-1]),8,2)) #+os.listdir(path)[-2:]
for i,n in enumerate(os.listdir(path)[:-1]):#+os.listdir(path)[-2:]
    string = str(open(path+n, 'r').read())
    string = string.split('{')[1][1:-3]
    dots = string.split('\n')
    dots = [[float(d.split(' ')[0]),float(d.split(' ')[1])] for d in dots]
    y1[i]=dots
del dots, i, n
#%% Image part

path =  'Photo\\E\\'
X1 = []
for i in os.listdir(path)[:-3]: #+os.listdir(path)[-2:]
    X1 += [io.imread(path+i)]
#face_im = io.imread(path)
del i
#%% functions


def display_frog(X, y):
    for i, p in enumerate(X):
        xi = y[i,:,0]
        yi = y[i,:,1]
        plt.imshow(p)
        plt.scatter(xi,yi)
        plt.axis('off')   # Don't display image axes
        plt.show()
    del xi, yi, i, p


def crude_tp(X, y):
    """
    transform into standard pose (mostly)
    using only flip
    """
    print('Commence crude alignment', end='')
    from skimage import transform
    crop_y = y
    tp_X = []
    tp_y = np.zeros(np.shape(crop_y))
    for i,p in enumerate(X):
        
        print('\nImage',i,': ',end='')
        x = crop_y[i,:,0]
        y = crop_y[i,:,1]
        
        if abs(crop_y[i,0,0]-crop_y[i,4,0])>abs(crop_y[i,0,1]-crop_y[i,4,1]): #x_head_butt > y_head_butt (normally should be y)
            print('rotate', end=' ')
            p=p.transpose(1,0,2)
            tmp = x
            x = y
            y = tmp
            del tmp
    
        if y[0]-y[4] > 0: #head lower than butt
            print('flip', end=' ')
            p = transform.rotate(p, 180)
            x = np.shape(p)[1]-x
            y = np.shape(p)[0]-y
        
        tp_X += [p]
        tp_y[i,:,0]=x
        tp_y[i,:,1]=y
    print('\n')
    return tp_X, tp_y


def fine_tp(X, y):
    """
    fine tune rotation using angle rotate
    """
    print('Commence fine tuning alignment', end='')
    from skimage import transform
    def centre(x1,x2,x3,x4):
        """
        finding center of gravity of 4 points
        input xn = (xn,yn)
        x1 = top
        x2 = left foreleg
        x3 = bottom
        x4 = right foreleg
        """
        x1,y1 = x1
        x2,y2 = x2
        x3,y3 = x3
        x4,y4 = x4
        if x3-x1!=0 and x4-x2!=0:
            mtb = (y3-y1)/(x3-x1) # slope top-bottom
            ctb = y3-mtb*x3       # intercept top-bottom
            mlr = (y4-y2)/(x4-x2)
            clr = y4-mlr*x4
            xc = (clr-ctb)/(mtb-mlr)
            yc = mtb*xc + ctb
            return (xc,yc)
        elif x3-x1==0:
            mlr = (y4-y2)/(x4-x2)
            clr = y4-mlr*x4
            yc = x3*mlr + clr
            return (x3, yc)
        elif x4-x2==0:
            mtb = (y3-y1)/(x3-x1)
            ctb = y3-mtb*x3
            yc = x4*mtb + ctb
            return (x4, yc)
        else:
            return None
    
    def rotate(origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.
    
        The angle in degree but need to transform to radian later within function.
        """
        ox, oy = origin
        px, py = point
        angle = np.deg2rad(angle)
    
        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy
    
    tp_y = y
    ft_X = []
    ft_y = np.zeros(np.shape(tp_y))
    for i,p in enumerate(X):
        print('\nImage',i,': ',end='')
        x = tp_y[i,:,0]
        y = tp_y[i,:,1]
        pivot = centre((x[0],y[0]),(x[2],y[2]),(x[4],y[4]),(x[6],y[6]))
        theta = np.arctan(abs(pivot[0]-x[0])/abs(pivot[1]-y[0])) #radian output
        theta = np.rad2deg(theta) #degree
        if x[0]<pivot[0]: #since rotation perform counterclockwise if top slightly left compare to center the need clock-wise = 360-counter-degree
            theta = 360 - theta
        p = transform.rotate(p,theta,center=pivot)
        x,y = rotate(pivot,(x,y),360-theta)
        print('rotate around',pivot,'counter-clockwise',theta,'degree', end='')
        #print(theta, np.deg2rad(theta),(x[0],y[0]),pivot)
        ft_X += [p]
        ft_y[i,:,0], ft_y[i,:,1] = x, y
    print('\n')
    return ft_X, ft_y

def sample_tp(X, y, scale_x=2.3, scale_yt=1.5, scale_yb=1.5):
    """
    Crop and down sample to standard size
    """
    print('Commence downsampling')
    from skimage import transform
    tp_y = y
    sam_X = []
    sam_y = np.zeros(np.shape(tp_y))
    for i,p in enumerate(X):
        print('\nImage',i,': ',end='')
        x = tp_y[i,:,0]
        y = tp_y[i,:,1]
        cen = (np.mean(x),np.mean(y))
        y1, y2 = max(0,int(cen[1]-scale_yt*abs(cen[1]-y[0]))), min(int(cen[1]+scale_yb*abs(cen[1]-y[4])),np.shape(p)[0])
        x1, x2 = max(0,int(cen[0]-scale_x*abs(cen[0]-x[3]))), min(int(cen[0]+scale_x*abs(cen[0]-x[5])), np.shape(p)[1])
        p = p[y1:y2, x1:x2]
        x = x-x1
        y = y-y1
        cen = (cen[0]-x1,cen[1]-y1)
        print('resizing from', np.shape(p),'to ',end='')
        new_p = np.zeros((max(np.shape(p)),max(np.shape(p)),3))
        offset_x = int((max(np.shape(p)) - np.shape(p)[1])/2)
        offset_y = int((max(np.shape(p)) - np.shape(p)[0])/2)
        new_p[offset_y:offset_y+np.shape(p)[0], offset_x:offset_x+np.shape(p)[1]] = p
        del p, x1, x2, y1, y2
        x = (x+offset_x)*512/np.shape(new_p)[1]
        y = (y+offset_y)*512/np.shape(new_p)[0]
        cen = ((cen[0]+offset_x)*512/np.shape(new_p)[1],(cen[1]+offset_y)*512/np.shape(new_p)[0])
        new_p = transform.resize(new_p, (512, 512, 3), anti_aliasing=True)
        sam_X += [new_p]
        sam_y[i,:,0] = x
        sam_y[i,:,1] = y
        print(np.shape(new_p))
    return sam_X, sam_y

def save_to_json(path,X, y):
    data = []
    for i, p in enumerate(X):
        data += [(p.tolist(), y[i,:,:].tolist())]
    import json
    print("Started writing list data into a json file")
    with open(path+".json", "w") as fp:
        json.dump(data, fp)
        print("Done writing JSON data into .json file")

def load_from_json(path):
    import json
    with open(path+'.json', 'rb') as fp:
        data = json.load(fp)
    X_e = []
    y_e = np.zeros((len(data),np.shape(data[0][1])[0],np.shape(data[0][1])[1]))
    for i,d in enumerate(data):
        X_e += [np.array(d[0])]
        y_e[i,:,:] = np.array(d[1])
    return X_e, y_e

def save_im(X,name,path='.'):
    for i,p in enumerate(X):
        im = (p * 255).astype(np.uint8)
        io.imsave(path+name[i][:-4]+'.jpg',im)
        
def save_pt(y, name, path='.'):
    for i,n in enumerate(y):
        np.savetxt(path+name[i][:-4]+'.csv', n)




#%% flow of processing and save the processed image
X, y = crude_tp(X, y)
X, y = fine_tp(X, y)
X1, y1 = sample_tp(X1, y1)
display_frog(X, y)
save_im(X, os.listdir('Photo\\E\\')[:-3]+os.listdir('Photo\\E\\')[-2:],'Photo\\E\\Processed\\')
save_pt(y, os.listdir('Label\\E\\')[:-3]+os.listdir('Label\\E\\')[-2:], 'Label\\E\\Processed\\')

#%% Neural Network part

import torch
import UNet3
from torchvision import transforms
from scipy import ndimage
from skimage.transform import downscale_local_mean
from skimage import transform
import torchvision.transforms as T

# ============ Utility functions ======================

def make_gauss_blob(im_size, x,y,blob_width=10):
    im=np.zeros(im_size)
    im[x,y]=1
    im = ndimage.gaussian_filter(im, sigma=blob_width)
    return im/im.max()

def convert_image_to_tensor(img):
    t_img=transforms.ToTensor()(img)
    t=torch.tensor((),dtype=torch.float32)
    t_in=t.new_ones((1,t_img.shape[0],t_img.shape[1],t_img.shape[2]))
    t_in[0]=t_img
    return t_in
    
def show_image(image,title):
    plt.imshow(image,cmap='gray')
    #plt.axis('off')   # Don't display image axes
    plt.title(title)
    plt.show()

def show_response_image(model,image,title,device):
    # Apply model to image and display the result
    y=model(convert_image_to_tensor(image).to(device)).to('cpu')
    plt.imshow(y.detach().numpy()[0][0],cmap='gray')
    plt.axis('off')   # Don't display image axes
    plt.title(title)
    plt.show()

def show_response_heat_map(model,image,title,device):
    # Apply model to image and display the result as heat map
    y=model(convert_image_to_tensor(image).to(device)).to('cpu')
    plt.imshow(image,cmap='gray',alpha=0.5)
    plt.imshow(y.detach().numpy()[0][0],cmap='plasma',alpha=0.5)
    plt.axis('off')   # Don't display image axes
    plt.title(title)
    plt.show()
    
def show_8_responses_heat_map(model,image,theta,point,spinee,title,device):
    # Apply model to image and display the result as heat map
    y=model(T.functional.rotate(convert_image_to_tensor(image).to(device), theta)).to('cpu')
    
    if point:
        points = []
        for i,result in enumerate(y.detach().numpy()[0]):
            # Find the position of the peak (maximum)
            # np.argmax() returns index as single integer
            # np.unravel_index converts that into the (i,j) position
            max_p=np.unravel_index(np.argmax(result, axis=None), result.shape)    
            plt.text(max_p[1],max_p[0],str(i),color="red")
            points += [[max_p[1],max_p[0]]]
        
        if spinee:
            a,b,spine = find_frog_spine_line(points)
            y_fit = [a+b*x for y,x in spine]
            x = [x for y,x in spine]
            plt.plot(y_fit,x)
            theta = theta_from_slope(b, points)
            image=transform.rotate(image, theta,center = points[0],resize=True)    
    else:
        image=transform.rotate(image, theta,center = None,resize=False)
    plt.imshow(image,cmap='gray',alpha=0.5)
    y = np.sum(y.detach().numpy()[0],axis=0)
    #y= transform.rotate(y,theta,center = points[0],resize=True)
    plt.imshow(y,cmap='plasma',alpha=0.5)
    
    #plt.axis('off')   # Don't display image axes
    plt.title(title)
    plt.show()

def show_n_responses_heat_map(model,n,image,theta,title,device):
    # Apply model to image and display the result as heat map
    y=model(T.functional.rotate(convert_image_to_tensor(image).to(device), theta)).to('cpu')
    plt.imshow(transform.rotate(image, theta),cmap='gray',alpha=0.5)
    plt.imshow(y.detach().numpy()[0][n],cmap='plasma',alpha=0.5)
    plt.axis('off')   # Don't display image axes
    plt.title(title)
    plt.show()
    
def show_image_and_peak(model,image,title,device):
    # Apply model to image. Display image and peak response point
    y=model(convert_image_to_tensor(image).to(device)).to('cpu')
    result=y.detach().numpy()[0][0]
    
    # Find the position of the peak (maximum)
    # np.argmax() returns index as single integer
    # np.unravel_index converts that into the (i,j) position
    max_p=np.unravel_index(np.argmax(result, axis=None), result.shape)    
    plt.plot(max_p[1],max_p[0],"x",color="red")


    plt.imshow(image,cmap='gray')
    plt.axis('off')   # Don't display image axes
    plt.title(title)
    plt.show()
    
def show_image_and_8_peaks(model,image,title,device):
    # Apply model to image. Display image and peak response point
    y=model(convert_image_to_tensor(image).to(device)).to('cpu')
    
    for i,result in enumerate(y.detach().numpy()[0]):
        # Find the position of the peak (maximum)
        # np.argmax() returns index as single integer
        # np.unravel_index converts that into the (i,j) position
        max_p=np.unravel_index(np.argmax(result, axis=None), result.shape)
        plt.plot(max_p[1],max_p[0],'x',color="red")
        plt.text(max_p[1],max_p[0],str(i),color="red")


    plt.imshow(image,cmap='gray')
    plt.axis('off')   # Don't display image axes
    plt.title(title)
    plt.show()
    
def get_8_points(model,image):
    # Apply model to image. Display image and peak response point
    y=model(convert_image_to_tensor(image).to(device)).to('cpu')
    return [np.unravel_index(np.argmax(result, axis=None), result.shape) for result in y.detach().numpy()[0]]

def find_frog_spine_line(points):
    spine_points = [tuple(np.average([points[0],tuple(np.average([points[1],points[7]],axis=0))],axis=0)),
                    tuple(np.average([points[2],points[6]],axis=0)),
                    tuple(np.average([points[4],tuple(np.average([points[3],points[5]],axis=0))],axis=0))
                    ]
    n = len(spine_points)
    y,X = np.transpose(spine_points)
    
    xbar = sum(X)/len(X)
    ybar = sum(y)/len(y)
    numer = sum([xi*yi for xi,yi in zip(X, y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2
    b = numer / denum
    a = ybar - b * xbar
    return a,b, spine_points

def theta_from_slope(b,points):
    if b>0:
        if points[0][1]<points[4][1]:
            print('b>0 and head to the left')
            return 360-(90-np.rad2deg(np.arctan(b)))
        else:
            print('b>0 and head right')
            return 180-(90-np.rad2deg(np.arctan(b)))
    elif b<0:
        if points[0][1]<points[4][1]:
            print('b<0 and head to the left')
            return 180+(90-abs(np.rad2deg(np.arctan(b))))
        else:
            print('b>0 and head to the right')
            return abs(np.rad2deg(np.arctan(b)))
    elif b==0:
        if points[0][1]>points[4][1]:
            return 90 #depend on which side facing
        else:
            return 270
        
def process_full_image(model,image,scale,device):
    y=model(convert_image_to_tensor(transform.rescale(image, scale, channel_axis=2)).to(device)).to('cpu')
    y = y.detach().numpy()[0]
    points = [np.unravel_index(np.argmax(result, axis=None), result.shape) for result in y]
    a,b,spine = find_frog_spine_line(points)
    theta = theta_from_slope(b, points)
    print(points,theta,b)
    image = transform.rotate(image,theta,resize=True,center=points[0])
    y = transform.rescale(y,1/scale,channel_axis=0)
    new_y=np.zeros((8,image.shape[0],image.shape[1]))
    for i in range(len(y)):
        new_y[i,:,:] = transform.rotate(y[i],theta,resize=True,center=points[0])
    print(new_y[0].shape,image.shape)
    y = new_y
    del new_y
    points = [np.unravel_index(np.argmax(result, axis=None), result.shape) for result in y]
    """for i,p in enumerate(points):
        plt.text(p[1],p[0],str(i))"""
    points = np.transpose(points)
    
    #cropping
    avg_x = np.average(points[0])
    avg_y = np.average(points[1])
    print(points,np.average(points[1]),np.average(points[0]))

    size_x = max([np.sqrt(x**2+y**2) for x,y in zip((points[0]-avg_x),(points[1]-avg_y))]) *2.5
    size_y = size_x
    x1, x2 = int(avg_y-size_x/2), int(avg_y+size_x/2)
    y1, y2 = int(avg_x-size_y/2+size_y/6), int(avg_x+size_y/2+size_y/6)
    plt.imshow(np.sum(y,axis=0)[y1:y2,x1:x2],cmap='plasma',alpha=0.5)
    plt.imshow(image[y1:y2,x1:x2,:],alpha=0.5)
    #plt.scatter(avg_y-y1,avg_x-x1,color='red')
    #plt.scatter(points[1]-y1,points[0]-x1,color='blue')
    plt.show()
    
    #return transform.resize(image[y1:y2,x1:x2,:], (512, 512, 3), anti_aliasing=True)
    show_8_responses_heat_map(model, transform.resize(image[y1:y2,x1:x2,:], (512, 512, 3), anti_aliasing=True), 0, 'test', device)
    
    
os.chdir("C:\\Users\\Eedrobup\\Downloads\\Frog photos")
#%% Point part

order = ['B','D','A','E']

y=np.zeros((0,8,2))
for folder in order:
    path = f'Label\\{folder}\\Processed\\'
    name = os.listdir(path)
    
    tmp_y = y
    y = np.zeros((len(name)+len(tmp_y),8,2))
    y[:len(tmp_y),:,:]=tmp_y
    for i,n in enumerate(name):
        y[i+len(tmp_y),:,:] = np.genfromtxt(path+n, delimiter=' ')
    del  i, n, tmp_y



#%% Image part

order = ['B','D','A','E']
X = []
for folder in order:
    path =  f'Photo\\{folder}\\Processed\\'
    name = os.listdir(path)
    #name.remove('Processed')
    for i in name:
        X += [io.imread(path+i)]
#face_im = io.imread(path)
del order, folder
#%% flip the wrong image

for i in range(len(X)):
    if y[i,1,0]<y[i,7,0] and y[i,2,0]<y[i,6,0] and y[i,3,0]<y[i,5,0]:
        print(i,'not flipped')
    else:
        print('flip both things')
        #flip image
        X[i] = np.fliplr(X[i]).copy()
        
        #flip points
        for j in range(8):
            y[i,j,0] = 512 - y[i,j,0]





#%%

# ============ Main ======================

# Load in an image

blob_width=10
#t_in = []
target_img = np.zeros((len(y),8,512,512))
for i,p in enumerate(X):
    #t_in += [convert_image_to_tensor(p)]
    #tmp = np.zeros((8,512,512))
    #k=0
    for j in range(0,8):
        #if j in [0,4]:
        #tmp[j,:,:] = make_gauss_blob(p.shape[:2], int(y[i,j,1]), int(y[i,j,0]),blob_width)
        target_img[i,j,:,:] = make_gauss_blob(p.shape[:2], int(y[i,j,1]), int(y[i,j,0]),blob_width)
    #target_img += [tmp]
    #plt.imshow(p,alpha=0.5)
    #plt.imshow(target_img[i],alpha=0.5,cmap='plasma')
    #plt.axis('off')   # Don't display image axes
    #plt.show()
target_img = [np.transpose(p,(1,2,0)) for p in target_img]
del i, j, p#, tmp, k
#%%
n=213
plt.imshow(X[n],cmap='gray',alpha=0.5)
plt.imshow(np.sum(target_img[n],axis=2),cmap='plasma',alpha=0.5)
for i,p in enumerate(np.transpose(target_img[n],(2,0,1))):
    max_p=np.unravel_index(np.argmax(p, axis=None), p.shape)
    plt.text(max_p[1],max_p[0],str(i),color="red")
plt.show()
del i,p,max_p,n
#%% Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet3.UNet3(32,64,128,8).to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1e-3)

#%%
# Convert img to a tensor
#t_in = [convert_image_to_tensor(p).to(device) for p in X]
t_in = [convert_image_to_tensor(p) for p in X]
t_inn = t_in[0]
for i in range(1,len(t_in)):
    t_inn = torch.cat([t_inn,t_in[i]],dim=0)


# Create a tensor and copy target image to it
#t_out=t_in.new_ones(t_in.shape)
#t_out[0]=transforms.ToTensor()(target_img[0])
t_out = [convert_image_to_tensor(p) for p in target_img]
t_outt = t_out[0]
for i in range(1,len(t_out)):
    t_outt = torch.cat([t_outt,t_out[i]],dim=0)

#clear memory
del t_out, t_in

# Create a dataset from t_in and t_out
from sklearn import model_selection
from sklearn.utils import shuffle

train_data, test_data, train_labels, test_labels = model_selection.train_test_split(t_inn.to(device), t_outt.to(device), test_size= 0.2, random_state = 1)
train_data, val_data, train_labels, val_labels = model_selection.train_test_split(train_data, train_labels, test_size= 0.2, random_state = 1)

#t_inn, t_outt = shuffle(t_inn, t_outt)
#train_set = torch.utils.data.TensorDataset(t_inn.to(device), t_outt.to(device))
#del t_inn, t_outt, i

train_set = torch.utils.data.TensorDataset(train_data, train_labels)
val_set = torch.utils.data.TensorDataset(val_data, val_labels)
test_set = torch.utils.data.TensorDataset(test_data, test_labels)

#clear memory
del train_data, val_data, test_data, train_labels, val_labels, test_labels, t_inn, t_outt, i


#%%
#Setup Data loader
from torch.utils.data import DataLoader

loaders = {
    'train' : DataLoader(train_set, 
                                          batch_size=1, 
                                          shuffle=True, 
                                          num_workers=0
                                          ),
    'val' : DataLoader(val_set, 
                                          batch_size=1, 
                                          shuffle=True, 
                                          num_workers=0
                                          ),
    
    'test'  : DataLoader(test_set, 
                                          batch_size=1,
                                          shuffle=True, 
                                          num_workers=0
                                          ),
}



#%% ==== Train the UNet =====
import torchvision.transforms as T


n_its=10
patch_size0= 512-32 #int(t_in[0].shape[2] -32)
patch_size1= 512-32 #int(t_in[0].shape[3] -32)
#theta = np.random.uniform(0,359,len(loaders['train'])).astype(int)
print(f"Using patches of size {patch_size0:d} x {patch_size1:d}")
for it in range(n_its):
    total_step = len(loaders['train'])
    for i, (images, labels) in enumerate(loaders['train']):
        # Take a random crop from the original image
        x=np.random.randint(0,images.shape[2]-patch_size0)
        y=np.random.randint(0,images.shape[3]-patch_size1)
        
        patch = images[:,:,x:x+patch_size0,y:y+patch_size1]
        target_patch = labels[:,:,x:x+patch_size0,y:y+patch_size1]
        
        #rotation with slightly random subangle adjustment
        theta = np.random.choice(range(0,351,30))
        randang = np.random.randint(-15,15)
        patch = T.functional.rotate(patch,int(theta+randang))
        target_patch = T.functional.rotate(target_patch,int(theta+randang))
        
            #resolution adjustment training
        for size in [(480,480),(768,768),(384,384),(960,960),(240,240)][0:2]: #np.random.randint(0,2)
            if size != (480,480):    
                patch = T.functional.resize(patch, size,antialias=True)
                target_patch = T.functional.resize(target_patch, size,antialias=True)
            
            model.train()
            y_pred = model(patch)
            
            loss = loss_fn(y_pred, target_patch)
           
            # Zero the gradients before running the backward pass.
            model.zero_grad()
            
            # Compute gradients. 
            loss.backward()
            
            # Use the optimizer to update the weights
            optimizer.step()
        
        if (i%5 == 4):
            model.eval()
            with torch.no_grad():
                loss=loss_fn(model(images),labels)
            print("It: {0:3d} Full Loss: {1:.6f}".format(it+1, loss.item()))
        
        if (i%25 == 24):
            model.eval()
            with torch.no_grad():
                show_8_responses_heat_map(model=model, image = X[0], theta=0, point=True,spinee=False,#i/total_step*360,
                                title="U-Net Result after {0:d} its".format(it+1),device=device)
                #show_8_responses_heat_map(model=model, image=transform.rotate(transform.rescale(B101[0],1/8,channel_axis=2),np.random.randint(-45,45)), theta=0, point=True, spinee=False, title='result from 1/8 diff data', device=device)
                #show_8_responses_heat_map(model=model, image=transform.rotate(transform.rescale(B101[9],1/4,channel_axis=2),np.random.randint(-45,45)), theta=0, point=True, spinee=False, title='result from 1/4 diff data', device=device)
                #show_8_responses_heat_map(model=model, image=transform.rotate(transform.rescale(B101[0],1/2,channel_axis=2),np.random.randint(-45,45)), theta=0, point=True, spinee=False, title='result from 1/2 diff data', device=device)

print("Done")
#%%
torch.save(model, './unet_first_layer')
#model = torch.load('./unetABDE').to(device)
#%%
show_8_responses_heat_map(model=model, image = transform.rescale(X1[0], 1/4, channel_axis=2), theta=0,#i/total_step*360,
                    title="U-Net Result",device=device)

show_image_and_8_peaks(model=model, image=transform.rescale(X1[0], 1/4, channel_axis=2), title='result', device=device)

#%%
plt.imshow(np.transpose(patch.to('cpu').detach().numpy()[0],(1,2,0)),alpha=0.5)
plt.imshow(np.sum(target_patch.to('cpu').detach().numpy()[0],axis=0),alpha=0.5)
plt.show()
#%%
# Apply the model and look at the result
show_response_image(model,X[0],"U-Net Result (on training image)")
#%%
# Run on a different image
show_response_image(model,X[7],"U-Net Result (on test image)",device=device)
show_image_and_peak(model,X[7],"U-Net Peak on Test image",device=device)
#%% load new set of photo
path='./Froggy ID pictures/B1/B1.05/'
B101=[]
name = os.listdir(path)
for i in name:
    B101 += [io.imread(path+i)]
#face_im = io.imread(path)
del i, name

show_8_responses_heat_map(model=model, image = transform.rescale(B101[10], 1/4, channel_axis=2), theta=0,#i/total_step*360,
                    title="U-Net Result",device=device)

show_image_and_8_peaks(model=model, image=transform.rescale(B101[1], 1/4, channel_axis=2), title='result', device=device)
