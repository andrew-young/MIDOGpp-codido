#!/usr/bin/env python

import argparse
from pathlib import Path
import os
import boto3
import zipfile
import shutil
import openslide
import logging
from logging import FileHandler
import time
import sys

class Codido:
	
	def __init__(self,arguments,root):
		
		self.root=root
		self.parser = argparse.ArgumentParser()
		self.parser.add_argument("--input", help="input")
		self.parser.add_argument("--output", help="output")
		self.parser.add_argument("--codido", help="running on codido")
		for arg,help in arguments:
			self.parser.add_argument("--"+arg, help=help)
		self.args = self.parser.parse_args()
		self.file=None
		

		
			
		#self.input_folder="."
		
		
		
		
		
		if self.args.codido == 'False' or  self.args.codido is None:
			cwd = os.getcwd()
			print(cwd)
			if cwd=="/app":
				self.root=cwd+"/"

			self.input_folder_path=self.root+"inputs/"
			self.output_folder_path =self.root+"outputs"
			self.copylocalinputs()
		else:
			self.root="./"
			#os.chdir("./app/")
			#self.input_folder_path=os.path.join(os.sep, 'app', 'inputs/')
			#self.output_folder_path = os.path.join(os.sep, 'app', 'outputs')
			self.input_folder_path="/app/inputs/"
			self.output_folder_path = "/app/outputs"
			self.input_file_path = os.path.join(self.input_folder_path, self.args.input.split('_SPLIT_')[-1])
			print(self.input_file_path)
			
			s3 = boto3.client('s3')
			s3.download_file(os.environ['S3_BUCKET'], self.args.input,  self.input_file_path)
			
	def logging(self,logfilefolder=None):
		if logfilefolder is not None:
			timestr = time.strftime("%Y%m%d-%H%M%S")
			self.logfilefolder=logfilefolder+"/log-"+timestr+".txt"
			self.file = open(self.logfilefolder, "w")
		else:
			self.logfilefolder=None
			self.file = None
			
	def write(self, data):
		if data is not None:
			if self.file is not None:
				self.file.write(str(data))
			print(data)	
					
	def copylocalinputs(self):
		file=self.getinputfile()
		if file is None:
			for folder_name, subfolders, filenames in os.walk(self.root+'localinputs/'):
				for filename in filenames:
					file_path=folder_name+"/" + filename 
					basename=os.path.basename(file_path)
					self.input_file_path=self.input_folder_path+"/"+basename
					print(file_path)
					print(self.input_file_path)
					shutil.copyfile(file_path,self.input_file_path)
					return
		
				
		
	def svstopng(self,file_name):
		img=openslide.OpenSlide(file_name)
		print('img read')
		print(img.dimensions)
		img=img.read_region((0,0),level=0,size=img.dimensions)
		print('img read done')	
		basename=Path(file_name).stem
		out_image_path=self.input_folder_path + basename+'.png'
		os.remove(file_name)
		img.save(out_image_path) #create png file
		#os.remove(file_name) #remove svs file
		file_name = out_image_path
		return file_name
	
	def getinputfile(self):#return first file found in inputs folder
		for folder_name, subfolders, filenames in os.walk(self.input_folder_path):
			for filename in filenames:
				file_path=folder_name+"/" + filename 
				return file_path
			
		return None
	
	def getinputfiles(self):#return first file found in inputs folder
		filelist=[]
		for folder_name, subfolders, filenames in os.walk(self.input_folder_path):
			for filename in filenames:
				file_path=filename 
				filelist.append(file_path)
		return filelist
		
	def unzip(self,filepath=None,dst=None):
		
		if filepath is None:
			filepath=self.input_file_path
			
		if dst is None:
			dst=self.input_folder_path
		
		f=Path(filepath)
		if f.suffix==".zip":
			print("unzip "+filepath+" to "+dst)	
			with zipfile.ZipFile(filepath, 'r') as zip_ref:
				zip_ref.extractall(dst)
	
	def uploadfiles(self):
		zip_name = self.output_folder_path + '.zip'
		print(zip_name)
		with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
			for folder_name, subfolders, filenames in os.walk(self.output_folder_path):
				for filename in filenames:
					file_path = os.path.join(folder_name, filename)
					zip_ref.write(file_path, arcname=os.path.relpath(file_path, self.output_folder_path))
		print('zip file created')

		file_stats = os.stat(zip_name)
		
	

		if self.args.codido == 'True':
			
			#config = TransferConfig(multipart_chunksize=200000)
			s3 = boto3.client('s3')
			#s3.upload_file(zip_name, os.environ['S3_BUCKET'], args.output, Config=config)
			s3.upload_file(zip_name, os.environ['S3_BUCKET'], self.args.output)		
	
	def cleanupimages(self,path=None):
		if path is None:
			path=self.input_folder_path
			
		# delete files moved from input folder to test folder
		for folder_name, subfolders, filenames in os.walk(path):
			for filename in filenames:
				file_path=folder_name+"/" + filename 
				os.unlink(file_path)
				

