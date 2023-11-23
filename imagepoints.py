# -*- coding: utf-8 -*-
"""
Data structure of image with its associated landmark points

@author: Pubordee Aussavavirojekul (eedrobup)
"""

import numpy as np
from skimage import io,color,transform
from skimage.util import crop
import json
import pickle

import landmarkdetector

class ImPo:
    """Object for a set of an image and its landmark points"""
    def __init__(self, frog_name:str, image_name:str, image: np.array, points: np.array) -> None:
        """loop reading images and points in main not here"""
        self.name = image_name
        self.frog = frog_name
        self.X = image
        self.y = points
        self.scale = None
        self.bounding_box = None
    
    def rotate(self, origin, point, angle):
        """
        Rotate a set of points counterclockwise by a given angle around a given origin.
        The angle in degree but need to transform to radian later within function.
        """
        ox, oy = origin
        px, py = point
        angle = np.deg2rad(angle)
    
        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy
    
    def rotate_resize(self, origin: tuple, point: list, angle: float, width, height):
        """
        Rotate a set of points counterclockwise by a given angle around a given origin.
        The angle in degree but need to transform to radian later within function.
        """
        ox, oy = origin
        px, py = point
        angle = np.deg2rad(angle)

        corners = [(0,0),(width,0),(0,height),(width, height)]
        dx = [np.cos(angle) * (x - ox) - np.sin(angle) * (y - oy) for x,y in corners]
        dy = [np.sin(angle) * (x - ox) + np.cos(angle) * (y - oy) for x,y in corners]
        
        qx = abs(min(dx)) + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = abs(min(dy)) + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy
    
    def get_name(self):
        return self.name
    
    def get_image(self):
        return self.X
    
    def get_points(self):
        return self.y
    
    def get_frog(self):
        return self.frog
    
    def set_boundary(self, vector:np.array) -> None:
        """set focus boundary to fetch a portion of image out"""
        pass
    
    def get_focused_impo(self, image = None, points = None, boundary = None):
        image = image if image is not None else self.X
        points = points if points is not None else self.y
        boundary = boundary if boundary is not None else self.bounding_box
        if boundary is None:
            raise TypeError("boundary cannot be None. Please give a boundary or use set_boundary method to setup a bounding box")
        else:
            x_min, y_min = boundary[0]
            x_max, y_max = boundary[1]
            points[:,0] -= x_min
            points[:,1] -= y_min
            return image[int(y_min):int(y_max),int(x_min):int(x_max),:], points
    
    def resample_impo(self, width:int, height:int, prev_ratio:bool=True):
        self.scale = [x1/x0 for x0,x1 in zip(self.X.shape[0:1], [height, width])]
        return [transform.rescale(self.X, scale=np.mean(self.scale) if prev_ratio==True else self.scale, channel_axis=2, anti_aliasing=True), #image
                [[x/np.mean(self.scale), y/np.mean(self.scale)] if prev_ratio==True else [x/self.scale[1], y/self.scale[0]] for x, y in self.y]] #points
    
    #TODO: need to complete this function for this commit
    def extract_patch(self,size)->np.array:
        """extract resampled patch surrounding by landmarks points to the specified size"""
        pass
    
    def save_pickle(self,path:str):
        """save this object instance to a file in provided directory
        
        Args:
            path (str): directory and name to save
        """
        with open(path, 'w') as f:
            pickle.dump(self, f)
    
    def save_json(self, path: str):
        """save image and its points to provided directory

        Args:
            path (str): directory without name (use impo name to save)
        """
        if path[-5:]==".json":
            raise Exception("Please provide only directory not with name")
        path = path+self.name if path[-1:]=="/" else path+"/"+self.name
        with open(path, 'w') as f:
            json.dump({'name': self.name,
                       'frog': self.frog,
                       'image': self.X.tolist(),
                       'points': self.y.tolist()}, f)

#TODO: inherit all parent methods with super().method()
class ManualImPo(ImPo):
    """Child class for an image-points pair where points are human labelled (ground truth)"""
    def __init__(self, frog_name:str, image: np.array, points: np.array) -> None:
        super().__init__(frog_name, image, points)
        self.pivot = np.mean(points, axis=0)
        self.theta = np.rad2deg(np.arctan2(points[0][0]-self.pivot[0],points[0][1]-self.pivot[1]))
    
    def get_std_image(self):
        """return image with frog in standard pose"""
        return transform.rotate(self.X,self.theta,center=self.pivot, resize=True)
    
    def get_std_points(self):
        """return coordinate of points of standard image"""
        return self.rotate_resize(origin=self.pivot, point=self.y, angle=self.theta, width=self.X.shape[1], height=self.X.shape[0])
    
    def get_std(self):
        """return both image in standard orientation and associated landmark points"""
        return transform.rotate(self.X,self.theta,
                                center=self.pivot,
                                resize=True), self.rotate_resize(origin=self.pivot, point=self.y,angle=self.theta,
                                                                   width=self.X.shape[1], height=self.X.shape[0])

#TODO: inherit all parent methods with super().method()
class GenImPo(ImPo):
    """Child class for an image-points pair where points are fetched from UNet3"""
    def __init__(self, image: np.array, model:landmarkdetector.model) -> None:
        #TODO: fix this get point function parameters
        points = model.get_points_from_image()
        super().__init__(image, points)
        self.layer = []

    def augment_points(self):
        """point augmentation to create better sense of whereabout of frog in image"""
        pass
        
#np.loadtxt('label\\E\\E9.09.pts', skiprows=3, max_rows=8)