import SimpleITK
import os
import yaml
from pandas import DataFrame
from evalutils import DetectionAlgorithm
from evalutils.validators import (
	UniquePathIndicesValidator,
	UniqueImagesValidator,
)
from pathlib import Path
from SlideRunner.dataAccess.database import Database
from MyMitosisDetection import MyMitosisDetection
import torch
from utils.nms_WSI import  nms, nms_patch
import json

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
		
