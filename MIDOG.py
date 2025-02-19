#22:34
import SimpleITK
import torch
import logging
import warnings
import math

import argparse
import zipfile

import os
import cv2
import shutil
import joblib
#import matplotlib as mpl
#import matplotlib.pyplot as plt
import numpy as np
import json

import pickle

import skimage

import openslide

from pandas import DataFrame


import torchvision.utils

import shutil
import argparse
from PIL import Image
import zipfile
from boto3.s3.transfer import TransferConfig
import boto3
import uuid
import pandas as pd
from Codido import Codido

from Mitosisdetection import Mitosisdetection

import scipy
#from scipy import scipy.signal.convolve
		
from pathlib import Path



#####################################################################################


class MIDOG():
	def __init__(self):
		arglist=[("mpp","area for mitosis density") , ("mm2photspot","area for mitosis density") , ("threshold","threshold for confidence core (0.5-1)")]
		self.codido=Codido(arglist,root="/mnt/hdd/docker/MIDOGpp-codido/")
		self.args=self.codido.args
		if self.args.codido == 'False' or  self.args.codido is None:
			self.codido.logging("/mnt/hdd/docker/MIDOGpp-codido/logging/")
			cwd = os.getcwd()
			print(cwd)
			if cwd=="/app":
				self.root=cwd+"/"
			else:
				self.root="/mnt/hdd/docker/MIDOGpp-codido/"
			self.mpp=0.25
			self.mm2pfov=1.
			self.threshold=0.5
			self.test_folder=self.root+'images/test/'
			self.input_folder=self.root+"inputs/"
			self.output_folder=self.root+"outputs/"
			
		else:
			self.root="./"
			self.mpp=float(self.args.mpp)
			self.mm2pfov=float(self.args.mm2photspot)
			self.threshold=float(self.args.threshold)
			#print(self.threshold)
			self.test_folder='./images/test/'
			self.input_folder="./inputs/"
			self.output_folder="./outputs/"
		os.makedirs(self.test_folder, exist_ok=True)
		
		self.codido.cleanupimages(self.output_folder)
		self.codido.cleanupimages(self.test_folder)
		self.codido.cleanupimages(self.input_folder)
		if self.args.codido == 'False' or  self.args.codido is None:
			self.codido.copylocalinputs()
			
			
		print(os.getcwd())
		
		self.mitosiscountdic={'Imagename': [], 'Mitotic figure count': [],"Mitotic figure density in Hotspot(mitotic figures per mm^2)":[],"Hotspot Size (mm^2)":[],"Image Area":[]}
		self.f2=None
		self.mitosistotal=0
		self.mitosiscountlist=[]
		self.nimages=0
		self.resultboxesdic={}
	
	def downsample(self,image, resize_factor):

		original_CT = SimpleITK.ReadImage(image,SimpleITK.sitkInt32)
		dimension = original_CT.GetDimension()
		reference_physical_size = np.zeros(original_CT.GetDimension())
		reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(original_CT.GetSize(), original_CT.GetSpacing(), reference_physical_size)]
		
		reference_origin = original_CT.GetOrigin()
		reference_direction = original_CT.GetDirection()

		reference_size = [round(sz/resize_factor) for sz in original_CT.GetSize()] 
		reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]

		reference_image = SimpleITK.Image(reference_size, original_CT.GetPixelIDValue())
		reference_image.SetOrigin(reference_origin)
		reference_image.SetSpacing(reference_spacing)
		reference_image.SetDirection(reference_direction)

		reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))
		
		transform = SimpleITK.AffineTransform(dimension)
		transform.SetMatrix(original_CT.GetDirection())

		transform.SetTranslation(np.array(original_CT.GetOrigin()) - reference_origin)
	  
		centering_transform = SimpleITK.TranslationTransform(dimension)
		img_center = np.array(original_CT.TransformContinuousIndexToPhysicalPoint(np.array(original_CT.GetSize())/2.0))
		centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
		centered_transform = SimpleITK.Transform(transform)
		centered_transform.AddTransform(centering_transform)

		# SimpleITK.Show(SimpleITK.Resample(original_CT, reference_image, centered_transform, SimpleITK.sitkLinear, 0.0))
		
		return SimpleITK.Resample(original_CT, reference_image, centered_transform, SimpleITK.sitkLinear, 0.0)
    
    	
	def createboundingboximage(self,result_boxes,input_image_file_path):
		imagename=os.path.basename(input_image_file_path)
		print(input_image_file_path)
		path=Path(input_image_file_path)
		print("./"+str(input_image_file_path))
		print(self.test_folder+imagename)
		img=Image.open(self.test_folder+imagename)
		#img=openslide.OpenSlide(self.test_folder+imagename)
		#width, height = img.dimensions
		img = img.convert('RGB')
		print(img)
		box=[]
		col=[]
		box2=[]
		col2=[]
		labels=[]
		smallbox=[]
		smallcol=[]
		smallbox2=[]
		smallcol2=[]
		smalllabels=[]
		dim=list(img.size)
		dim=[dim[1],dim[0]]
		print(dim)
		#print(type(input_image))
		#print(imagename)
		#print(input_image)
		
		#dim=list(input_image.GetSize())#y,x

		maxdim=max(dim[0],dim[1])
		smalldim=[dim[0],dim[1]]
		smallscale=1
		if maxdim >1000:
			smallscale=1000/maxdim
			smalldim[0]=(int)(smallscale*dim[0])
			smalldim[1]=(int)(smallscale*dim[1])
		print("resizeing")
		#smallimg=skimage.transform.resize(img,tuple(smalldim))
		#downsample=self.downsample(self.test_folder+imagename,smallscale)
		print("resized")
		#img = SimpleITK.GetArrayFromImage(input_image)
		#print("got array")
		#img=img[:,:,0:3]
		#img=np.transpose(img,(2,0,1))
		#img = torch.from_numpy(img)
		
		smallimg=np.transpose(img,(2,0,1))
		smallimg = torch.from_numpy(smallimg)
		smallresize=torchvision.transforms.Resize(size=smalldim)
		smallimg = smallresize(smallimg)
		#print(smalldim)
		pixelmap=torch.zeros(1,1,dim[0]//4,dim[1]//4, device=torch.device('cuda'))
		print([dim[0]//4,dim[1]//4])
		a=int(1000/self.mpp*math.sqrt(self.mm2pfov))
		dimfov=[a,a]#
		
		#if fov is larger than image decrease fov
		if dimfov[0]>dim[0]:
			dimfov[0]=dim[0]
		if dimfov[1]>dim[1]:
			dimfov[1]=dim[1]
			
		hotboxw=dimfov[1]
		hotboxh=dimfov[0]
		self.mm2pfov=dimfov[0]*dimfov[1]*self.mpp*self.mpp/1000/1000
		img=np.transpose(img,(2,0,1))
		img = torch.from_numpy(img)
		boxwidth=7
		for i, detection in enumerate(result_boxes):
			# our prediction returns x_1, y_1, x_2, y_2, prediction, score -> transform to center coordinates
			x_1, y_1, x_2, y_2, prediction, score = detection
			x=int(x_1+x_2)//2
			y=int(y_1+y_2)//2
			print([x,y])
			threshold2=(1+self.threshold)/2
			if score >self.threshold:
				pixelmap[0,0,y//4,x//4]=pixelmap[0,0,y//4,x//4]+1

		
		if len(result_boxes)>=1:
			w=dimfov[1]//2
			h=dimfov[0]//2
			#dim=(pixelmap.size(2)//4,pixelmap.size(3)//4)
			#resiz=torchvision.transforms.Resize((dim), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
			#pixelmap3=resiz(pixelmap)
			#print(pixelmap3.shape)
			pool = torch.nn.AvgPool2d([hotboxh//4,hotboxw//4],stride=1)
			#pixelmap = pixelmap.cuda()
			print("pool")
			pixelmap2=pool(pixelmap)
			#pixelmap2 = pixelmap2.cpu()
			print(pixelmap2.shape)
			dim2=(pixelmap2.size(2),pixelmap2.size(3))
			print(dim2)
			pixelmap2=pixelmap2
			x=torch.argmax(pixelmap2,keepdim =True)
			print(x)
			x=x.cpu()
			hotbox_y,hotbox_x=np.unravel_index(x,dim2)
		
			print((hotbox_x,hotbox_y))
			nummitosis=(pixelmap2[0,0,hotbox_y,hotbox_x]*hotboxh*hotboxw//4//4).cpu().numpy()
			print(nummitosis)
			hotbox_x=hotbox_x*4
			hotbox_y=hotbox_y*4
			hotbox_x=hotbox_x+a//2
			hotbox_y=hotbox_y+a//2
			print("img2e")

		
			
			#if int(nummitosis)==1:
			#	y,x=np.unravel_index(np.argmax(pixelmap),pixelmap2.shape)

			self.mm2pfov=w*h*self.mpp*self.mpp/1000/1000
			#dim=(img.shape[1],img.shape[2])#y,x
			print(dim)
			if hotbox_x-w-boxwidth<0:
				
				hotbox_x=w
				w=w-boxwidth
			if hotbox_y-h-boxwidth< 0:
				
				hotbox_y=h
				h=h-boxwidth

			if hotbox_x+w+boxwidth>dim[1]:
				
				hotbox_x=dim[1]-w
				w=w-boxwidth
			if hotbox_y+h+boxwidth>dim[0]:
				
				hotbox_y=dim[0]-h	
				h=h-boxwidth
			
			self.mm2pfov=2*2*w*h*self.mpp*self.mpp/1000/1000
			print(self.mm2pfov)
			self.density=(nummitosis/self.mm2pfov)
			#print(img.shape)
			hotboximg=torchvision.transforms.functional.crop(img,hotbox_y-h,hotbox_x-w,2*(h),2*(w) )
			print("created hotbox")
			#hotboximg=torch.concat(hotboximg,axis=0)
			#print([hotbox_x-w-boxwidth,hotbox_y-h-boxwidth,2*(w+boxwidth),2*(h+boxwidth)])
			#print(smalldim)
			
			#smallimg=smallresize(img)
			
			print("created lowres image")
			print([2*w,2*h])
				
			for i, detection in enumerate(result_boxes):
				# our prediction returns x_1, y_1, x_2, y_2, prediction, score -> transform to center coordinates
				x_1, y_1, x_2, y_2, prediction, score = detection
				x=int(x_1+x_2)//2
				y=int(y_1+y_2)//2
				#print((x,y))
				pixelmap[0,0,y//4,x//4]=pixelmap[0,0,y//4,x//4]+1
				threshold2=(1+self.threshold)/2
				if score >self.threshold:
					
					labels.append(' %.3f' %score)
					smalllabels.append(' %.3f' %score)

					# draw bounding box and fill color 
					#box.append([x_1-5, y_1-5, x_2+5, y_2+5])
					if x_1> hotbox_x-w and x_2< hotbox_x+w and y_1> hotbox_y-h and y_2< hotbox_y+h:
						print([x_1-15 -hotbox_x+w, y_1-15 -hotbox_y+h , x_2+15 -hotbox_x+w, y_2+15 -hotbox_y+h])
						box2.append([x_1-15 -hotbox_x+w, y_1-15 -hotbox_y+h , x_2+15 -hotbox_x+w, y_2+15 -hotbox_y+h])
						box.append([x_1-15-boxwidth -hotbox_x+w, y_1-15-boxwidth -hotbox_y+h, x_2+15+boxwidth -hotbox_x+w, y_2+15+boxwidth -hotbox_y+h])
						col2.append((0,0,0))
						if score > threshold2:
							x=int( (score-threshold2)/(1-threshold2)*255)
							col.append((255-x,x,0))
						else:
							x=int( (score-self.threshold)/(threshold2-self.threshold)*255 )
							col.append((0,x,255-x))
					smallbox2.append([int(x_1*smallscale)-5, int(y_1*smallscale)-5, int(x_2*smallscale)+5, int(y_2*smallscale)+5])
					smallbox.append([int(x_1*smallscale)-5-1, int(y_1*smallscale)-5-1, int(x_2*smallscale)+5+1, int(y_2*smallscale)+5+1])
		
					smallcol2.append((0,0,0))
					if score > threshold2:
						x=int( (score-threshold2)/(1-threshold2)*255)
						smallcol.append((255-x,x,0))
					else:
						x=int( (score-self.threshold)/(threshold2-self.threshold)*255 )
						smallcol.append((0,x,255-x))
					#col.append((0,0,0))
			smallbox.append([int((hotbox_x-w-boxwidth)*smallscale), int((hotbox_y-h-boxwidth)*smallscale), int((hotbox_x+w+boxwidth)*smallscale), int((hotbox_y+h+boxwidth)*smallscale)])
			smallbox2.append([int((hotbox_x-w)*smallscale),int((hotbox_y-h)*smallscale),int((hotbox_x+w)*smallscale),int((hotbox_y+h)*smallscale)])
			smalllabels.append(' %.3f' %self.density)
			smallcol.append((0,0,0))
			smallcol2.append((0,0,0))
			
			smallbox=np.asarray(smallbox)
			smallbox=smallbox.astype(int)
			smallbox = torch.tensor(smallbox)
			smallimg = torchvision.utils.draw_bounding_boxes(smallimg, boxes=smallbox,labels=None, width=1, colors=smallcol,  fill=False)#
			smallbox2=np.asarray(smallbox2)
			smallbox2=smallbox2.astype(int)
			smallbox2 = torch.tensor(smallbox2) 
			smallimg = torchvision.utils.draw_bounding_boxes(smallimg, boxes=smallbox2,labels=smalllabels, width=1, colors=smallcol2,  fill=False, font_size=30)#
			
			
			box=np.asarray(box)
			box=box.astype(int)
			box = torch.tensor(box)
			#print(box)
			#print(hotboximg.shape)
			hotboximg = torchvision.utils.draw_bounding_boxes(hotboximg, boxes=box,labels=None, width=boxwidth, colors=col,  fill=False)#
			box2=np.asarray(box2)
			box2=box2.astype(int)
			box2 = torch.tensor(box2) 
			hotboximg = torchvision.utils.draw_bounding_boxes(hotboximg, boxes=box2,labels=labels, width=1, colors=col2,  fill=False, font_size=30)#
			
			#hotboximg = hotboximg.permute(1, 2,0)/255.,
			#print(hotboximg)
			#print(img.shape)
			#print(hotboximg.shape)
			#torchvision.utils.save_image(hotboximg,fp=self.output_folder+'/hotbox_'+imagename+'.png')
			#print("img5")
					#torchvision.utils.save_image(smallimg,fp=self.output_folder+'/lowres_'+imagename+'.png')
			hotboximg=hotboximg.numpy()
			hotboximg=np.transpose(hotboximg,(1,2,0))
			im=Image.fromarray(hotboximg)
			
			out_image_path=self.output_folder+'/hotbox_'+imagename+'.png'
			print("saving: "+out_image_path)
			im.save(out_image_path)
			print("saved")
		else:
			self.density=0
			

		
		smallimg=smallimg.numpy()
		smallimg=np.transpose(smallimg,(1,2,0))
		im=Image.fromarray(smallimg)
		
		out_image_path=self.output_folder+'/lowres_'+imagename+'.png'
		print("saving: "+out_image_path)
		im.save(out_image_path)
		print("saved")
		
		
	def countmitotisis(self,result_boxes,input_image_file_path):
		imagename=os.path.basename(input_image_file_path)
		#img=openslide.OpenSlide(input_image_file_path)
		#width, height = img.dimensions
		img=Image.open(self.test_folder+imagename)
		dim=list(img.size)
		width, height = dim
		print(self.renamedic2)
		ogimagename=self.renamedic2[imagename]
		pixels=width*height
		areaimage=pixels*self.mpp*self.mpp/1000/1000
		areafov=self.mm2pfov
		fovpimage=areaimage/areafov
		
		mitosiscount=len(result_boxes)
		self.mitosiscountlist.append(mitosiscount)

		self.mitosiscountdic["Imagename"].append(ogimagename)
		self.mitosiscountdic["Mitotic figure count"].append(mitosiscount)
		self.mitosiscountdic["Mitotic figure density in Hotspot(mitotic figures per mm^2)"].append(self.density)
		self.mitosiscountdic["Hotspot Size (mm^2)"].append(self.mm2pfov)
		self.mitosiscountdic["Image Area"].append(areaimage)
		
		#print(mitosiscount)
		
	#called by Mitosisdetection.process_case
	def process_case(self,result_boxes,input_image_file_path,input_image):
		imagename=os.path.basename(input_image_file_path)
		ogimagename=self.renamedic2[imagename]
		split=os.path.splitext(ogimagename)
		noextension=split[0]
		
		self.resultboxesdic[input_image_file_path]=result_boxes
		
	
	def process_case2(self):
	
		for input_image_file_path in self.resultboxesdic:
			result_boxes=self.resultboxesdic[input_image_file_path]
			
			imagename=os.path.basename(input_image_file_path)
			ogimagename=self.renamedic2[imagename]
			split=os.path.splitext(ogimagename)
			noextension=split[0]
			
			self.createboundingboximage(result_boxes,input_image_file_path)
			self.countmitotisis(result_boxes,input_image_file_path)
	
	#run model
	def inference(self,directory):  
		#every folder in directory is assumed to contain a model run all of them (should be 1)
		for root, dirs, files in os.walk(directory):
			for dir in dirs:
				with open(os.path.join(directory, dir, "files", "wandb-summary.json"), 'r') as f:
					data = json.load(f)
				detection = Mitosisdetection(os.path.join(directory, dir, "files"))
				detection.codido=self
				# loads the image(s), applies DL detection model & saves the result
				#self.codido.write("Evaluating", dir)
				#detection.move_validation_slides(test=True)
				detection.process()
			break
		
	
	def mitosis_count_averagecsv(self):
		mitotic_figure_count_average=np.sum(self.mitosiscountlist)/self.nimages
		mitotic_figure_count_standard_deviation=np.std(self.mitosiscountlist)
		d = {'﻿number_of_images': [self.nimages], 'mitotic_figure_count_average': [mitotic_figure_count_average],'mitotic_figure_count_standard_deviation':[mitotic_figure_count_standard_deviation]}
		df = pd.DataFrame(data=d)
		self.codido.write(df)
		df.to_csv(self.output_folder+"/mitosis_count_average.csv", index=False)
	
	def mitosis_countcsv(self):
		df = pd.DataFrame(data=self.mitosiscountdic)
		self.codido.write(df)
		df.to_csv(self.output_folder+"/mitosiscount.csv", index=False)
	
		

	#renames files and return dictionary containing origional filnames, convert svs files to png
	def uniquefilenames(self):
		renamedic={}#uuid filnames to unique og filenames
		self.renamedic2={}#numbered filenames to og filenames.
		filei=1;
		self.codido.write(os.getcwd())
		self.codido.write(self.input_folder)
		#first rename to unique files names to avoid name clashes
		for folder_name, subfolders, filenames in os.walk(self.input_folder):
			for filename in filenames:
				file_path=folder_name+"/" + filename 
				print(file_path)
				split=os.path.splitext(filename)
				extension=split[1]
				unique_filename = str(uuid.uuid4())+extension
				renamedic[unique_filename]=filename
				old_file = os.path.join(folder_name, filename)
				new_file = os.path.join(folder_name, unique_filename)
				os.rename(old_file, new_file)
		
		print("asf")
		#rename files to numbers as is desired.
		for unique_filename in renamedic:
			

			split=os.path.splitext(unique_filename)
			extension=split[1]
			basename=split[0]
			#print(extension)
			old_file = os.path.join(self.input_folder, unique_filename)
			


			
			print(extension)
			
			if extension==".svs":
				old_file=self.codido.svstotiff(old_file)
				filename=str(filei)+".tif"
				new_file = os.path.join(self.test_folder, filename)
				#os.rename(old_file, new_file)
				shutil.copyfile(old_file, new_file)
				print("copy")
			else:
				filename=str(filei)+extension
				new_file = os.path.join(self.test_folder, filename)
				self.codido.cleanupimages(self.test_folder)
				shutil.copyfile(old_file, new_file)
				#os.rename(old_file, new_file)
			
			self.codido.write(new_file)
			self.renamedic2[filename]=renamedic[unique_filename]
			
			filei=filei+1
		self.nimages=filei-1
			
		return self.renamedic2
			
		
	def run(self):

		path=self.codido.getinputfile()
		
		self.codido.unzip(path)
		
		self.renamedic2=self.uniquefilenames()
		self.codido.write(self.renamedic2)
		
		warnings.filterwarnings("ignore")
		
		os.chdir(self.root)
		self.inference('wandb')
		
		self.process_case2()
		self.mitosis_count_averagecsv()
		self.mitosis_countcsv()
		
		# create zip with all the saved outputs
		self.codido.uploadfiles()
		#self.codido.cleanupimages(self.test_folder)
		#self.codido.cleanupimages(self.output_folder)

