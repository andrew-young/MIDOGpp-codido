import SimpleITK
from pathlib import Path
from queue import Queue
from tqdm import tqdm
import logging
import warnings


import argparse
import zipfile

import os
import cv2
import shutil
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from utils.detection_helper import create_anchors, process_output, rescale_boxes
from utils.nms_WSI import  nms, nms_patch
import os
import pickle
import yaml
from object_detection_fastai.models.RetinaNet import RetinaNet
from fastai.vision.learner import create_body
from fastai.vision import models
from SlideRunner.dataAccess.database import Database
from pandas import DataFrame
import torch
from evalutils import DetectionAlgorithm
from evalutils.validators import (
	UniquePathIndicesValidator,
	UniqueImagesValidator,
)
import torchvision.utils
import json
import shutil
import argparse
from PIL import Image
import boto3
import zipfile
from boto3.s3.transfer import TransferConfig
import boto3
import uuid
import pandas as pd

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
			self.input_file_path = self.getinputfile()
			#print(os.listdir("."))
			self.mpp=0.25
			self.mm2pfov=0.25

		self.test_folder='./images/test/'#os.path.dirname(self.input_file_path)
	
	def getinputfile(self):#erturn first file found in inputs folder
		for folder_name, subfolders, filenames in os.walk('./inputs'):
			for filename in filenames:
				file_path=folder_name+"/" + filename 
				return file_path


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


	#renames files and return dictionary containing origional filnames
	def uniquefilenames(self):
		renamedic={}
		self.renamedic2={}
		filei=0;
		
		#rename to unique files names to avoid name clashes
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
			filei=filei+1
			filename=str(filei)+extension
			self.renamedic2[filename]=renamedic[unique_filename]
			split=os.path.splitext(unique_filename)
			extension=split[1]
			
			old_file = os.path.join(folder_name, unique_filename)
			new_file = os.path.join(folder_name, filename)
			os.rename(old_file, new_file)
		return self.renamedic2

	def process_case(self,result_boxes,input_image_file_path,input_image):
		imagename=os.path.basename(input_image_file_path)
		ogimagename=self.renamedic2[imagename]
		split=os.path.splitext(ogimagename)
		noextension=split[0]
		img=input_image
		width, height = img.GetSize()
		pixels=width*height
		areaimage=pixels*self.mpp*self.mpp/1000/1000
		areafov=self.mm2pfov
		fovpimage=areaimage/areafov
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
		#box = box.unsqueeze(0) 
		#print(box)
		img = torchvision.utils.draw_bounding_boxes(img, box, width=10, colors=col,   fill=False)#
		mitosiscount=len(result_boxes)
		self.mitosiscountlist.append(mitosiscount)
		
		self.f2.write(ogimagename+", ")
		self.f2.write(str(mitosiscount)+", ")
		self.f2.write(str(mitosiscount/fovpimage)+"\n")
		
		
		print(mitosiscount)
		img=img.numpy()
		img=np.transpose(img,(1,2,0))
		im=Image.fromarray(img)
		out_image_path=self.output_folder_path +'/boundingboxes'+noextension+'.png'
		im.save(out_image_path)
	
	
	def inference(self,directory):  
	
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
	
	def run(self):
		


		self.movefiles()
			
		self.renamedic2=self.uniquefilenames()
				
		self.f2= open("./outputs/mitosiscount.csv","w")
		self.f2.write("Imagename, Mitotic figure count,Mitotic figure count per fov(field of view="+str(self.mm2pfov)+"mm^2)\n")

		warnings.filterwarnings("ignore")
		mpl.rcParams["figure.dpi"] = 300  # for high resolution figure in notebook
		print("test 3")
		mpl.rcParams["figure.facecolor"] = "white"  # To make sure text is visible in dark mode
		plt.rcParams.update({"font.size": 5})

		self.inference('wandb')
		
		self.f2.close()
		self.mitosis_count_averagecsv()
		
		# create zip with all the saved outputs
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
		# upload
		for folder_name, subfolders, filenames in os.walk(self.test_folder):
			for filename in filenames:
				file_path=self.test_folder + filename 
				os.unlink(file_path)

		if args.codido == 'True':
			import boto3
			#config = TransferConfig(multipart_chunksize=200000)
			s3 = boto3.client('s3')
			#s3.upload_file(zip_name, os.environ['S3_BUCKET'], args.output, Config=config)
			s3.upload_file(zip_name, os.environ['S3_BUCKET'], args.output)


		





class MyMitosisDetection:
	def __init__(self, path, config, detect_threshold, nms_threshold):
		with open('statistics_sdata.pickle', 'rb') as handle:
			statistics = pickle.load(handle)
		tumortypes = config["data"]["value"]["tumortypes"].split(",")
		self.mean = np.array(np.mean(np.array([value for key, value in statistics['mean'].items() if tumortypes.__contains__(key)]),axis=(0, 1)), dtype=np.float32)
		self.std = np.array(np.mean(np.array([value for key, value in statistics['std'].items() if tumortypes.__contains__(key)]),axis=(0, 1)), dtype=np.float32)

		# network parameters
		self.detect_thresh = detect_threshold
		self.nms_thresh = nms_threshold
		encoder = create_body(models.resnet18, True, -2)
		scales = [float(s) for s in config["retinanet"]["value"]["scales"].split(",")]
		ratios = [config["retinanet"]["value"]["ratios"]]
		sizes = [(config["retinanet"]["value"]["sizes"], config["retinanet"]["value"]["sizes"])]
		self.model = RetinaNet(encoder, n_classes=2, n_anchors=len(scales) * len(ratios),sizes=[size[0] for size in sizes], chs=128, final_bias=-4., n_conv=3)
		self.path_model = os.path.join(path, "bestmodel.pth")
		self.size = config["data"]["value"]["patch_size"]
		self.batchsize = config["data"]["value"]["batch_size"]

		self.anchors = create_anchors(sizes=sizes, ratios=ratios, scales=scales)
		self.device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

	def load_model(self):
		if torch.cuda.is_available():
			print("Model loaded on CUDA")
			self.model.load_state_dict(torch.load(self.path_model)['model'])
		else:
			print("Model loaded on CPU")
			self.model.load_state_dict(torch.load(self.path_model, map_location='cpu')['model'])

		self.model.to(self.device)

		logging.info("Model loaded. Mean: {} ; Std: {}".format(self.mean, self.std))
		return True

	def process_image(self, input_image):
		self.model.eval()
		n_patches = 0
		queue_patches = Queue()
		img_dimensions = input_image.shape

		image_boxes = []
		# create overlapping patches for the whole image
		for x in np.arange(0, img_dimensions[1], int(0.9 * self.size)):
			for y in np.arange(0, img_dimensions[0], int(0.9 * self.size)):
				# last patch shall reach just up to the last pixel
				if (x+self.size>img_dimensions[1]):
					x = img_dimensions[1]-512

				if (y+self.size>img_dimensions[0]):
					y = img_dimensions[0]-512

				queue_patches.put((0, int(x), int(y), input_image))
				n_patches += 1


		n_batches = int(np.ceil(n_patches / self.batchsize))
		for _ in tqdm(range(n_batches), desc='Processing an image'):

			torch_batch, batch_x, batch_y = self.get_batch(queue_patches)
			class_pred_batch, bbox_pred_batch, _ = self.model(torch_batch)

			for b in range(torch_batch.shape[0]):
				x_real = batch_x[b]
				y_real = batch_y[b]

				cur_class_pred = class_pred_batch[b]
				cur_bbox_pred = bbox_pred_batch[b]
				cur_patch_boxes = self.postprocess_patch(cur_bbox_pred, cur_class_pred, x_real, y_real)
				if len(cur_patch_boxes) > 0:
					image_boxes += cur_patch_boxes

		return np.array(image_boxes)

	def get_batch(self, queue_patches):
		batch_images = np.zeros((self.batchsize, 3, self.size, self.size))
		batch_x = np.zeros(self.batchsize, dtype=int)
		batch_y = np.zeros(self.batchsize, dtype=int)
		for i_batch in range(self.batchsize):
			if queue_patches.qsize() > 0:
				status, batch_x[i_batch], batch_y[i_batch], image = queue_patches.get()
				x_start, y_start = int(batch_x[i_batch]), int(batch_y[i_batch])

				cur_patch = image[y_start:y_start+self.size, x_start:x_start+self.size] / 255.
				batch_images[i_batch] = cur_patch.transpose(2, 0, 1)[0:3]
			else:
				batch_images = batch_images[:i_batch]
				batch_x = batch_x[:i_batch]
				batch_y = batch_y[:i_batch]
				break
		torch_batch = torch.from_numpy(batch_images.astype(np.float32, copy=False)).to(self.device)
		for p in range(torch_batch.shape[0]):
			torch_batch[p] = transforms.Normalize(self.mean, self.std)(torch_batch[p])
		return torch_batch, batch_x, batch_y

	def postprocess_patch(self, cur_bbox_pred, cur_class_pred, x_real, y_real):
		cur_patch_boxes = []

		for clas_pred, bbox_pred in zip(cur_class_pred[None, :, :], cur_bbox_pred[None, :, :], ):
			modelOutput = process_output(clas_pred, bbox_pred, self.anchors, self.detect_thresh)
			bbox_pred, scores, preds = [modelOutput[x] for x in ['bbox_pred', 'scores', 'preds']]

			if bbox_pred is not None:
				# Perform nms per patch to reduce computation effort for the whole image (optional)
				to_keep = nms_patch(bbox_pred, scores, self.nms_thresh)
				bbox_pred, preds, scores = bbox_pred[to_keep].cpu(), preds[to_keep].cpu(), scores[
					to_keep].cpu()

				t_sz = torch.Tensor([[self.size, self.size]]).float()

				bbox_pred[:, :2] = bbox_pred[:, :2] - bbox_pred[:, 2:] / 2
				bbox_pred = rescale_boxes(bbox_pred, t_sz)

				for box, pred, score in zip(bbox_pred, preds, scores):
					y_box, x_box = box[:2]
					h, w = box[2:4]

					cur_patch_boxes.append(
						np.array([x_box + x_real, y_box + y_real,
								  x_box + x_real + w, y_box + y_real + h,
								  pred, score]))

		return cur_patch_boxes
		
		
		
		
		
		
		
		
		
		
class Mitosisdetection(DetectionAlgorithm):
	def __init__(self, path):
		self.codido=None
		# Read YAML file
		
		with open(os.path.join(path, "config.yaml"), 'r') as stream:
			self.config = yaml.safe_load(stream)
		super().__init__(
			validators=dict(
				input_image=(
					UniqueImagesValidator(),
					UniquePathIndicesValidator(),
				)
			),
			input_path = Path(os.path.join(self.config['files']['value']['image_path'],"test")),
			output_file = Path(os.path.join(path, "mitotic-figures.json"))
		)
		self.detect_thresh = 0.5
		self.nms_thresh = 0.4

		self.database = Database()
		#self.database.open(Path("databases/MultiDomainMitoticFigureDataset.sqlite"))
		self.database.open(Path("databases/MIDOG++.sqlite"))
		#self.uids = dict(self.database.execute('SELECT filename,uid from Slides').fetchall())
		self.gts = {}

		#####################################################################################
		# Note: As of MIDOG 2022, the format has changed to enable calculation of the mAP. ##
		#####################################################################################
		# Use NMS threshold as detection threshold for now so we can forward sub-threshold detections to the calculations of the mAP

		self.md = MyMitosisDetection(path, self.config, self.detect_thresh, self.nms_thresh)
		load_success = self.md.load_model()
		if load_success:
			print("Successfully loaded model.")

	def move_validation_slides(self, test):
		for slide in json.loads(self.config['x-validation']['valid']):
			if test:
				os.rename(os.path.join(self.config['files']['value']['image_path'], slide),
						  os.path.join(self._input_path, slide))
			else:
				os.rename(os.path.join(self._input_path, slide),
						  os.path.join(self.config['files']['value']['image_path'], slide))

	def gt_annotations(self, slideId, input_image):
		bboxes = []
		self.database.loadIntoMemory(slideId)
		for id, annotation in self.database.annotations.items():
			if len(annotation.labels) != 0 and annotation.deleted != 1:
				label = annotation.agreedClass
				if label == 1:  # labeled as MF
					coords = np.mean(annotation.coordinates, axis=0)
					world_coords = input_image.TransformContinuousIndexToPhysicalPoint([c for c in coords])
					bboxes.append([*tuple(world_coords), 0])
		return bboxes

	def save(self):
		with open(str(self._output_file), "w") as f:
			json.dump(dict(zip([c[1].loc['path'].name for c in self._cases['input_image'].iterrows()], self._case_results)), f)

	def process_case(self, *, idx, case):
		# Load and test the image for this case
		input_image, input_image_file_path = self._load_input_image(case=case)
		print(input_image_file_path)
		#print(type(input_image))
		#self.filei=self.filei+1
		#self.gts[input_image_file_path.name] = self.gt_annotations(self.uids[input_image_file_path.name], input_image)

		# Detect and score candidates
		result_boxes = self.predict(input_image=input_image)

	
  

		# transform this image to PIL image 
		
		self.codido.process_case(result_boxes,input_image_file_path,input_image)
		
		
		# Write resulting candidates to result.json for this case
		return dict(type="Multiple points", points=None, version={ "major": 1, "minor": 0 })

	def predict(self, *, input_image: SimpleITK.Image) -> DataFrame:
		# Extract a numpy array with image data from the SimpleITK Image
		#print(input_image)

		candidates = list()
		classnames = ['non-mitotic figure', 'mitotic figure']
		
		image_data = SimpleITK.GetArrayFromImage(input_image)
		#print(input_image)
		with torch.no_grad():
			result_boxes = self.md.process_image(image_data)

		# perform nms per image:
		print("All computations done, nms as a last step")
		result_boxes = nms(result_boxes, self.nms_thresh)
		return result_boxes
		



def main():
	codido=Codido()
	codido.run()

main()
