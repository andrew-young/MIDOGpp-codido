

import SimpleITK
import torch
import logging
import warnings


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

from Mitosisdetection import Mitosisdetection


parser = argparse.ArgumentParser()
parser.add_argument("--input", help="input")
parser.add_argument("--output", help="output")
parser.add_argument("--codido", help="running on codido")
parser.add_argument("--mpp", help="area for mitosis density")
parser.add_argument("--mm2pfov", help="area for mitosis density")

args = parser.parse_args()


		
class Codido:
	def __init__(self):
		self.f2=None
		self.mitosistotal=0
		self.mitosiscountlist=[]
		
		self.nimages=0
		if args.codido == 'True':

			self.input_folder_path=os.path.join(os.sep, 'app', 'inputs')
			self.output_folder_path = os.path.join(os.sep, 'app', 'outputs')
			os.makedirs(self.input_folder_path, exist_ok=True)
			os.makedirs(self.output_folder_path, exist_ok=True)


			s3 = boto3.client('s3')

			# downloads codido input file into the folder specified by input_folder_path
			self.input_file_path = os.path.join(self.input_folder_path, args.input.split('_SPLIT_')[-1])
			

			s3.download_file(os.environ['S3_BUCKET'], args.input, self.input_file_path)
			
			self.mpp=float(args.mpp)
			self.mm2pfov=float(args.mm2pfov)
			#output_folder_path=self.output_folder_path
		else:
			self.output_folder_path='./outputs'
			self.input_folder_path='./inputs'
			self.input_file_path = self.getinputfile()
			os.makedirs(self.input_folder_path, exist_ok=True)
			os.makedirs(self.output_folder_path, exist_ok=True)
			if self.input_file_path is None:
				print("inputs folder empty")
				return
			#print(os.listdir("."))
			self.mpp=0.25
			self.mm2pfov=0.25

		self.test_folder='./images/test/'#os.path.dirname(self.input_file_path)
		os.makedirs(self.test_folder, exist_ok=True)
		self.mitosiscountdic={'Imagename': [], 'Mitotic figure count': [],'Mitotic figure count per fov(field of view='+str(self.mm2pfov)+'mm^2)':[],"Image Area":[]}
		
		
	def getinputfile(self):#return first file found in inputs folder
		for folder_name, subfolders, filenames in os.walk('./inputs'):
			for filename in filenames:
				file_path=folder_name+"/" + filename 
				return file_path
		return None


	#moves files from input folder to test folder
	#extracts zip file
	def movefiles(self):
		filename=os.path.basename(self.input_file_path)
		split=os.path.splitext(filename)
		extension=split[1]
		print(extension)
		if extension==".zip":
			with zipfile.ZipFile(self.input_file_path, 'r') as zip_ref:
				zip_ref.extractall(self.test_folder)
			imagelist=[]
			for folder_name, subfolders, filenames in os.walk(self.test_folder):
				for filename in filenames:
					imagelist.append(filename)		
		else:
			shutil.copyfile(self.input_file_path,'./images/test/'+filename)	
			imagelist=[filename]
			

		print(self.input_file_path)
		print(imagelist)
		self.nimages=len(imagelist)


	def svstopng(self,file_name):
		img=openslide.OpenSlide(file_name)
		img=img.read_region((0,0),level=0,size=img.dimensions)
		out_image_path=self.test_folder + basename+'.png'
		img.save(out_image_path) #create png file
		os.remove(self.test_folder + unique_filename) #remove svs file
		file_name = out_image_path
		return file_name
	
	#renames files and return dictionary containing origional filnames, convert svs files to png
	def uniquefilenames(self):
		renamedic={}#uuid filnames to unique og filenames
		self.renamedic2={}#numbered filenames to og filenames.
		filei=1;
		
		#first rename to unique files names to avoid name clashes
		for folder_name, subfolders, filenames in os.walk(self.test_folder):
			for filename in filenames:
				file_path=folder_name+"/" + filename 
				split=os.path.splitext(filename)
				extension=split[1]
				unique_filename = str(uuid.uuid4())+extension
				renamedic[unique_filename]=filename
				old_file = os.path.join(folder_name, filename)
				new_file = os.path.join(folder_name, unique_filename)
				os.rename(old_file, new_file)
				
		#rename files to numbers as is desired.
		for unique_filename in renamedic:
			

			split=os.path.splitext(unique_filename)
			extension=split[1]
			basename=split[0]
			print(extension)
			old_file = os.path.join(folder_name, unique_filename)
			
			if extension==".svs":
				old_file=svstopng(old_file)
				extension=".png"
				
			tiles=self.checklargefiles(old_file)
			if len(tiles) >1 and tiles is not None:
				print("tiles "+str(len(tiles)))
				for x,y,img in tiles:
					filename=str(filei)+extension
					ogname=renamedic[unique_filename]
					split=os.path.splitext(ogname)
					extension=split[1]
					basename=split[0]
					
					self.renamedic2[filename]=basename+"_"+str(x)+"_"+str(y)+extension
					new_file = os.path.join(folder_name, filename)
					img.save(new_file)
					
					filei=filei+1
				os.unlink(old_file)
			else:		
				filename=str(filei)+extension
				self.renamedic2[filename]=renamedic[unique_filename]
				new_file = os.path.join(folder_name, filename)
				os.rename(old_file, new_file)
				filei=filei+1
			
		return self.renamedic2
	

	#return tiles of image data incase image is large.
	def checklargefiles(self,file):
		sizethreshold=3e7
		j=None
		with Image.open(file) as img:
			w,h=img.size
			print(w*h)
			
			if w*h>sizethreshold:
				
				for i in range(1,10):
					if w*h/i/i < sizethreshold:
						j=i
						break
			else:
				j=1
			if j is None:
				print("file too large")
				
			list=[]
			M=w/j
			N=h/j
			#img=img.numpy()
			print(img.size)
			img=img.convert("RGB")
			#data=img.getdata()
			#img = np.array(data).reshape(w, h, 3)
			#print(img.size)
			img=np.array(img)
			print(img.shape)
			if j is not None:
				for x in range(j):
					for y in range(j):
						list.append( (x,y, Image.fromarray(img[int(x*M):int(M*(x+1)) , int(y*N):int((y+1)*N),:])   )  )	
	
		return list
	
	def createboundingboximage(self,result_boxes,imagename,input_image):
		img = SimpleITK.GetArrayFromImage(input_image)
		img=np.transpose(img,(2,0,1))
		img = torch.from_numpy(img)
		
		box=[]
		col=[]
		
		for i, detection in enumerate(result_boxes):
			# our prediction returns x_1, y_1, x_2, y_2, prediction, score -> transform to center coordinates
			x_1, y_1, x_2, y_2, prediction, score = detection

			# draw bounding box and fill color 
			#box.append([x_1-5, y_1-5, x_2+5, y_2+5])
			box.append([x_1-10, y_1-10, x_2+10, y_2+10])
			col.append((255,0,0))
			#col.append((0,0,0))
			
		box=np.asarray(box)
		box=box.astype(int)
		box = torch.tensor(box) 
		img = torchvision.utils.draw_bounding_boxes(img, box, width=10, colors=col,   fill=False)#
		img=img.numpy()
		img=np.transpose(img,(1,2,0))
		im=Image.fromarray(img)
		out_image_path=self.output_folder_path +'/boundingboxes'+imagename+'.png'
		im.save(out_image_path)
		
	def countmitotisis(self,result_boxes,imagename,input_image):
		img=input_image
		width, height = img.GetSize()
		pixels=width*height
		areaimage=pixels*self.mpp*self.mpp/1000/1000
		areafov=self.mm2pfov
		fovpimage=areaimage/areafov
		
		mitosiscount=len(result_boxes)
		self.mitosiscountlist.append(mitosiscount)

		self.mitosiscountdic["Imagename"].append(imagename)
		self.mitosiscountdic["Mitotic figure count"].append(mitosiscount)
		self.mitosiscountdic["Mitotic figure count per fov(field of view="+str(self.mm2pfov)+"mm^2)"].append(mitosiscount/fovpimage)
		self.mitosiscountdic["Image Area"].append(areaimage)
		
		print(mitosiscount)
		
	#called by Mitosisdetection.process_case
	def process_case(self,result_boxes,input_image_file_path,input_image):
		imagename=os.path.basename(input_image_file_path)
		ogimagename=self.renamedic2[imagename]
		split=os.path.splitext(ogimagename)
		noextension=split[0]
		
		self.createboundingboximage(result_boxes,noextension,input_image)
		self.countmitotisis(result_boxes,ogimagename,input_image)
		
		
	
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
				print("Evaluating", dir)
				#detection.move_validation_slides(test=True)
				detection.process()
			break
		
	
	def mitosis_count_averagecsv(self):
		mitotic_figure_count_average=np.sum(self.mitosiscountlist)/self.nimages
		mitotic_figure_count_standard_deviation=np.std(self.mitosiscountlist)
		d = {'﻿number_of_images': [self.nimages], 'mitotic_figure_count_average': [mitotic_figure_count_average],'mitotic_figure_count_standard_deviation':[mitotic_figure_count_standard_deviation]}
		df = pd.DataFrame(data=d)
		print(df)
		df.to_csv(self.output_folder_path+"/mitosis_count_average.csv", index=False)
	
	def mitosis_countcsv(self):
		df = pd.DataFrame(data=self.mitosiscountdic)
		print(df)
		df.to_csv(self.output_folder_path+"/mitosiscount.csv", index=False)
		
	def run(self):
	
		self.movefiles()
			
		self.renamedic2=self.uniquefilenames()
		print(self.renamedic2)
	
		warnings.filterwarnings("ignore")
		

		self.inference('wandb')
	
		self.mitosis_count_averagecsv()
		self.mitosis_countcsv()
		
		# create zip with all the saved outputs
		self.uploadfiles()
		self.cleanup()

	def uploadfiles(self):
		zip_name = self.output_folder_path + '.zip'
		print(zip_name)
		with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
			for folder_name, subfolders, filenames in os.walk(self.output_folder_path):
				for filename in filenames:
					file_path = os.path.join(folder_name, filename)
					print('create zip file')
					print(file_path)
					print(self.output_folder_path)
					zip_ref.write(file_path, arcname=os.path.relpath(file_path, self.output_folder_path))
					print('zip file created')

		file_stats = os.stat(zip_name)
		print(file_stats.st_size)
		
	

		if args.codido == 'True':
			import boto3
			#config = TransferConfig(multipart_chunksize=200000)
			s3 = boto3.client('s3')
			#s3.upload_file(zip_name, os.environ['S3_BUCKET'], args.output, Config=config)
			s3.upload_file(zip_name, os.environ['S3_BUCKET'], args.output)

	def cleanup(self):
		# delete files moved from input folder to test folder
		for folder_name, subfolders, filenames in os.walk(self.test_folder):
			for filename in filenames:
				file_path=self.test_folder + filename 
				os.unlink(file_path)

def main():
	codido=Codido()
	codido.run()

main()
