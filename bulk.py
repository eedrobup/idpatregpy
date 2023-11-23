# -*- coding: utf-8 -*-
"""
Data structure of bulk imagepoints (resemble a tank of frogs)

@author: Pubordee Aussavavirojekul (eedrobup)
"""
import os
import datetime
import json

import numpy as np
import pandas as pd
from scipy import ndimage
from torch.utils.data import DataLoader
from sklearn import model_selection
from sklearn.utils import shuffle

import torch
import UNet3
from torchvision import transforms
from scipy import ndimage
from skimage.transform import downscale_local_mean
from skimage import transform
import torchvision.transforms as T

import imagepoints

std_col = ["utc_time_stamp","name","image","points","vdes","vec","im_name"]

class Bulk:
    def __init__(self, bulk_name:str, path:str) -> None:
        #set path
        path = path if path[-1]!="/" else path[:-1]
        
        if path[-1*(len(bulk_name)):]==bulk_name:
            self.path = path
        else:
            os.mkdir(path+"/"+bulk_name)
            self.path = path+"/"+bulk_name
        
        #load object from path if available
        if bulk_name+".json" in os.listdir(path):
            with open(path+"/"+bulk_name+".json",'r') as f:
                self.__dict__ = json.loads(f)
        else:
            self.bulk_name = bulk_name
        
        self.data = self.load_data(self.bulk_name)
        pass
    
    def load_data(self, name:str):
        """return pandas DF of data if present otherwise return blank dataframe with standard columns"""
        if not name+".csv" in os.listdir(self.path):
            return pd.DataFrame(columns=std_col)
        else:
            return pd.read_csv(self.path+"/"+name+".csv")
    
    def add_entity(self, frog_name, impo:imagepoints.ImPo):
        id = self.get_new_id()
        self.data.loc[id,"utc_time_stamp"] = datetime.datetime.utcnow()
        self.data.loc[id,"name"] = frog_name
        self.data.loc[id,"image"] = None
        self.data.loc[id,"vdes"] = None
        self.data.loc[id,"vec"] = None
        self.data.loc[id,"im_name"] = impo.get_name()
        if not self.path+"/"+impo.get_name()+".json" in os.listdir(self.path):
            impo.save_json(self.path)
        pass
    
    def get_new_id(self) -> int:
        return int(self.data.shape[0])
    
    def save_data(self) -> None:
        """save bulk data to csv"""
        self.data.to_csv(self.path+"/"+self.bulk_name+".csv")

#TODO: inherit all parent methods with super().method()
class TrainingBulk(Bulk):
    def __init__(self, bulk_name:str, path:str) -> None:
        super().__init__(bulk_name, path)
        self.input_size = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet3.UNet3(32,64,128,8).to(self.device)
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1e-3)
        self.loaders = None
        self.loaders_done = False
        self.loss_tracker = pd.DataFrame(columns=["train_loss","val_loss"])
        
    def add_entity(self, frog_name, impo:imagepoints.ManualImPo) -> None:
        """add a new entry of training image"""
        id = self.get_new_id()
        self.data.loc[id,"utc_time_stamp"] = datetime.datetime.utcnow()
        self.data.loc[id,"name"] = frog_name
        self.data.loc[id,"image"] = impo.get_std_image()
        self.data.loc[id,"points"] = impo.get_std_points()
        self.data.loc[id,"im_name"] = impo.get_name()
        if not self.path+"/"+impo.get_name()+".json" in os.listdir(self.path):
            impo.save_json(self.path)
        self.loaders_done = False
    
    def __generate_target_image(self, im_size:tuple, list_of_points:np.array,blob_width:int = 10)->np.array:
        """receive ground truth of landmark points (no. of points = number of layers) and return target masks"""
        def make_gauss_blob(im_size:tuple, x:int,y:int,blob_width:int=10)->np.array:
            im=np.zeros(im_size)
            im[x,y]=1
            im = ndimage.gaussian_filter(im, sigma=blob_width)
            return im/im.max()
        
        target_img = np.zeros((len(list_of_points),im_size[0],im_size[1]))
        for j in range(len(list_of_points)):
            target_img[j,:,:] = make_gauss_blob(im_size, int(list_of_points[j,1]), int(list_of_points[j,0]),blob_width)
        return np.transpose(target_img,(1,2,0))
        
    def __set_dataloader(self, images_in:torch.tensor, target_masks:torch.tensor,
                         train_val_test_ratio:tuple = (0.6,0.2,0.2),
                         batch_size:int = 32, random_state:int = 1,
                         num_workers=0)->None:
        """create dataloader for training process"""
        if sum(train_val_test_ratio)!=1:
            raise Exception("train_val_test_ration must sum up to 1")
        train_X, test_X, train_y, test_y = model_selection.train_test_split(images_in.to(self.device),
                                                                            target_masks.to(self.device),
                                                                            test_size= train_val_test_ratio[2],
                                                                            random_state = random_state)
        train_X, val_X, train_y, val_y = model_selection.train_test_split(train_X,
                                                                          train_y,
                                                                          test_size= train_val_test_ratio[1]/(1-train_val_test_ratio[2]),
                                                                          random_state = random_state)
        self.loaders = {'train' : DataLoader(torch.utils.data.TensorDataset(train_X, train_y), 
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=num_workers
                                ),
                        'val' : DataLoader(torch.utils.data.TensorDataset(val_X, val_y), 
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=num_workers
                                ),

                        'test'  : DataLoader(torch.utils.data.TensorDataset(test_X, test_y), 
                                batch_size=batch_size,
                                shuffle=True, 
                                num_workers=num_workers
                                ),
                        }
        self.loaders_done=True
    def convert_image_to_tensor(img):
        t_img=transforms.ToTensor()(img)
        t=torch.tensor((),dtype=torch.float32)
        t_in=t.new_ones((1,t_img.shape[0],t_img.shape[1],t_img.shape[2]))
        t_in[0]=t_img
        return t_in

    def train(self, max_iteration = 10, early_stop = None, random_crop:tuple =(32,32),
              list_of_resolution = [(480,480),(768,768),(384,384),(960,960),(240,240)]):
        #prepare input and output mask from data and create dataloader from it
        if self.loaders_done==False or (self.loaders is None):
            #if data is updated and loader has not yet updated
            images_in = torch.stack([self.convert_image_to_tensor(p) for p in self.data["image"]], dim=0)
            #FIXME: check how self.input_size is handled
            target_img = [self.__generate_target_image(self.input_size,pts) for pts in self.data["points"]]
            target_masks = torch.stack([self.convert_image_to_tensor(p) for p in target_img],dim=0)
            self.__set_dataloader(images_in, target_masks)
            del images_in, target_img, target_masks #clear memory
        
        #Model training part
        patch_size0 = self.input_size-random_crop[0]
        patch_size1 = self.input_size-random_crop[1]
        for it in range(max_iteration):
            #total_step = len(self.loaders['train'])
            train_los = []
            for i, (images, labels) in enumerate(self.loaders['train']):
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
                
                #random resolution adjustment training
                size = list_of_resolution[np.random.randint(0,len(list_of_resolution))]
                if size != (patch_size0,patch_size1):    
                    patch = T.functional.resize(patch, size,antialias=True)
                    target_patch = T.functional.resize(target_patch, size,antialias=True)
                
                self.model.train()
                y_pred = self.model(patch)
                
                loss = self.loss_fn(y_pred, target_patch)
            
                # Zero the gradients before running the backward pass.
                self.model.zero_grad()
                
                # Compute gradients. 
                loss.backward()
                
                # Use the optimizer to update the weights
                self.optimizer.step()
                
                # log train loss
                if i%5 == 4:
                    print(f"It: {it+1:3d} train Loss: {loss.item():.6f}")
                train_los += [loss.item()]
                
            #compare validation set
            self.model.eval()
            with torch.no_grad():
                val_loss=[]
                for images, labels in self.loaders['val']:
                    output = self.model(images)[0]
                    loss = self.loss_func(output, labels)
                    val_loss += [loss.item()]
                    
            self.loss_tracker.loc[it,"train_loss"] = sum(train_los)/len(train_los)
            self.loss_tracker.loc[it,"val_loss"] = sum(val_loss)/len(val_loss)
            
            #case early stopping
            val_index = self.loss_tracker.columns.get_loc("val_loss")
            if not early_stop is None:
                if sum(val_loss)/len(val_loss) > sum(train_los)/len(train_los):
                    if early_stop >= it:
                        continue
                    if min(self.loss_tracker.iloc[:-1*early_stop,val_index]) < min(self.loss_tracker.iloc[-1*early_stop:,val_index]):
                        break
                pass
            del train_los, val_loss, val_index
        #report with test set
        self.model.eval()
        with torch.no_grad():
            test_loss=[]
            for images, labels in self.loaders['test']:
                output = self.model(images)[0]
                loss = self.loss_func(output, labels)
                test_loss += [loss.item()]
        print(f"Training Finished\nThe final loss on the test set is {sum(test_loss)/len(test_loss):.6f}")
        pass
    
    #TODO:
    def set_image_size(self, size:tuple)->None:
        self.input_size = size
    
    def save_model(self, name:str):
        torch.save(self.model, './'+name)
        

#TODO: inherit all parent methods with super().method()
class DatabaseBulk(Bulk):
    def __init__(self, bulk_name:str, path:str):
        super().__init__(bulk_name, path)
        self.vec_descriptor="HOG"
        self.__archived = self.load_data(self.bulk_name+"_archived")
        
    def add_entity(self, frog_name, impo:imagepoints.GenImPo):
        id = self.get_new_id()
        self.data.loc[id,"utc_time_stamp"] = datetime.datetime.utcnow()
        self.data.loc[id,"name"] = frog_name
        self.data.loc[id,"image"] = None
        self.data.loc[id,"vdes"] = self.vec_descriptor
        self.data.loc[id,"vec"] = self.__vector_extraction(impo)
        self.data.loc[id,"im_name"] = impo.get_name()
        if not self.path+"/"+impo.get_name()+".json" in os.listdir(self.path):
            impo.save_json(self.path)
        pass
    
    def load_data(self, name: str):
        return super().load_data(name+"_db")
    
    def save_data(self) -> None:
        """save bulk data to csv"""
        self.data.to_csv(self.path+"/"+self.bulk_name+"_db.csv")
        self.__archived.to_csv(self.path+"/"+self.bulk_name+"_archived_db.csv")
    
    def __softmax(score:dict)->dict:
        """return a set of dictionary with softmax transform"""
        keys = list(score.keys())
        numerators = [np.exp(score[i]) for i in keys]
        denom = sum(numerators)
        return {key:num/denom for key,num in zip(keys,numerators)}
    
    def __get_distances(self, image_vec, dist_func):
        score = {}
        for id in self.data["name"].unique():
            score[id] = np.mean([dist_func(x,image_vec) for x in self.data[self.data["name"]==id]["vec"]])
        return self.__softmax(score)

    def __vector_extraction(self, impo: imagepoints.ImPo) -> np.array:
        """Vector descriptor extraction"""
        """ For python 3.10
        match self.vec_descriptor:
            case "HOG":
                #do HOG encoding
                print("encoding HOG")
            case _:
                print("default")"""
        if self.vec_descriptor=="LBH":
            pass
        else: #self.vec_descriptor=="HOG" and default
            return self.__HOG(impo)
        pass
    
    def __HOG(image:imagepoints.ImPo, bin=8, pixels_per_cell=(32,32), cells_per_block=(2,2)):
        """return feature vector encoded by Histogram of Oriented Gradients feature descriptor"""
        def __hist(mag, direc, histtick:np.array):
            hist = np.zeros(len(histtick)-1)
            bin = len(hist)
            width = max(histtick)/bin
            for i,rad in enumerate(direc):
                #calculate bin voting with bilinear interpolation and assign magnitude accordingly
                binthis = np.floor(rad/width-0.5)%bin
                binnext = (binthis+1)%bin
                hist[binthis] += mag[i]*((histtick[binnext+1]+histtick[binnext+1])/2-rad)/width
                hist[binnext] += mag[i]*(rad-(histtick[binthis]+histtick[binthis-1])/2)/width
            return hist
        
        def __norm(dmatrix:np.array)->np.array:
            epsilon = np.finfo(float).eps
            return dmatrix/np.sqrt(np.sum(dmatrix**2)+epsilon)

        image = image.get_image() #add extract patch function later
        
        # Basic gradient filters
        xgrad_filter=np.array([[-1,0,1]], dtype=np.float32)
        ygrad_filter=np.array([[-1],[0],[1]], dtype=np.float32)
        
        #calculate gradient in x and y axis
        gx = np.empty_like(image)
        gy = np.empty_like(image)
        for i in range(image.shape[0]):
            gx[i,:] = np.convolve(image[i:],xgrad_filter[0,:], mode='same')
        for i in range(image.shape[1]):
            gy[:,i] = np.convolve(image[:,i],ygrad_filter[:,0],mode='same')
        
        # Gradient magnitude and orientation calculation
        mag = np.sqrt(gx**2 + gy**2)
        ort = np.arctan2(gy,gx)
        del gx,gy
        
        #cell histogram calculation (non overlapping filter)
        orttick = np.linspace(0,2*np.pi,num=bin+1, endpoint=True)
        numcells = (np.ceil(image.shape[i]/float(pixels_per_cell[i])) for i in [0,1])
        cells = np.empty((numcells[0],numcells[1],bin))
        for by in range(numcells[0]):
            for bx in range(numcells[1]):
                cells[by,bx,:]=__hist(mag[by*pixels_per_cell[0]:min(image.shape[0],(by+1)*pixels_per_cell[0]),
                                          bx*pixels_per_cell[1]:min(image.shape[1],(bx+1)*pixels_per_cell[1])].reshape(-1),
                                      ort[by*pixels_per_cell[0]:min(image.shape[0],(by+1)*pixels_per_cell[0]),
                                          bx*pixels_per_cell[1]:min(image.shape[1],(bx+1)*pixels_per_cell[1])].reshape(-1),
                                      orttick)
        del numcells, orttick
        
        #block normalization (overlapping filter)
        numblocks = (cells.shape[i]-cells_per_block[i]+1 for i in [0,1])#(np.ceil(cells.shape[i]/float(cells_per_block[i])) for i in [0,1])
        blocks = np.empty((numblocks[0],numblocks[1],bin))
        for by in range(numblocks[0]):
            for bx in range(numblocks[1]):
                blocks[by,bx,:]=__norm(cells[by:min(cells.shape[0],by+cells_per_block[0]),
                                             bx:min(cells.shape[1],bx+cells_per_block[1]),:].reshape(-1))
        del numblocks
          
        #normalize the full HOG feature
        pos_clipping_threshold = 0.2 #clipping too large gradient leading to contrast independency
        HOG_des = __norm(np.apply_along_axis(func1d = lambda x:min(x,pos_clipping_threshold),
                                             axis = 0,
                                             arr = __norm(blocks.reshape(-1))))
                
        return HOG_des
    
    def lean_data() -> None:
        """ FIXME: NOT IMPLEMENTED IN THIS COMMIT
        put the unrepresenting images into the archive"""
        pass

    def im_match(self, impo:imagepoints.ImPo, method:str = "cosine") -> dict:
        """
        Check how the query image match any pattern within the group
        
        Parameters
        ----------
        impo : imagepoints.ImPo
            an image and landmark points surrounding region of interest of pattern
        method : str
            a distance method to use in matching algorithm (default is cosine distance)
        Returns
        -------
        score
            a dictionary of every entities in the group and matching score (using softmax function)
        """
        image = impo.extract_patch() #need attention!!!
        if method =="euclidean":
            return self.__get_distances(self.__HOG(image), lambda x,y: np.sqrt(np.sum(np.square(x-y))))
        elif method=="cosine":
            return self.__get_distances(self.__HOG(image),np.dot)
        else:
            return self.__get_distances(self.__HOG(image),np.dot)
    
    def hungarian_match(self, impo_list:list, method:str) -> dict:
        """FIXME:
        NOT IMPLEMENTED IN THIS COMMIT
        Check how the query images match any pattern within the group
        
        Parameters
        ----------
        impo : list of imagepoints.ImPo
            a list of pairs of image and landmark points
        method : str
            a distance method to use in matching algorithm (default is cosine distance)
        Returns
        -------
        pair of match : list of tuples
            a list of best matching pair of query images and identities within the group
        """
        pass