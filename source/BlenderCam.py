import bpy
import numpy as np
import math

class BlenderCam:
	"""Class to load camera metadata from a blender scene/file.

	Export data like FOV, position and orientation for the use in VoxelCast.
	"""
	def __init__(self, filepath):
		self.filepath = filepath
		bpy.ops.wm.open_mainfile(filepath=filepath)
		
		self.cameras = {}
		self.camdat = {}
		obj = bpy.data.objects#.keys()
		for i in obj:
			if i.type == "CAMERA":
				self.cameras[i.name] = i#bpy.data.objects[i]
				#print(self.cameras[i].data.angle)
				self.camdat[i.name] = self.projectionData(i.name)

	def getAllCams(self):
		return list(self.cameras.keys())

	def doesCamExist(self, name):
		if name not in self.getAllCams():
			raise ValueError(f"Camera with the name {name} is not found in loaded cameras. Loaded are {self.getAllCams()} from the file {self.filepath}")

	def revEulerTransform(self, alpha, beta, gamma):
		"""Returns a reverse transformation/rotation matrix for euler angles applied in the order alpha -> beta -> gamma
		
		:param alpha, beta, gamma: euler angles in radians
		:type alpha, beta, gamma: float
		"""
		# TODO: handle locking in
		p = [alpha, beta, gamma]
		cx, cy, cz = tuple([math.cos(i) for i in p])
		sx, sy, sz = tuple([math.sin(i) for i in p])

		m = np.array([
			[cy*cz, cy*sz, -sy],
			[sx*sy*cz-cx*sz, sx*sy*sz+cx*cz, sx*cy],
			[cx*sy*cz+sx*sz, cx*sy*sz-sx*cz, cx*cy]
		])

		invm = np.linalg.inv(m)
		return invm

	def getCamPosOri(self, name):
		"""Return the position as vector and orientation as transformation matrix of a camera

		:param name: name of the camera, which pos/ori should be returned
		:type name: str
		:return: tuple of (np.array<float> 1D, np.array<float> 2D)
		"""
		self.doesCamExist(name) # otherwise raise exception
		
		cam = self.cameras[name]
		loc = np.array(list(cam.location)).transpose()
		#print(loc)
		rot = tuple(cam.rotation_euler)
		m = self.revEulerTransform(*rot)
		print(f"{name}: {rot}")
		return loc, m

	def projectionData(self, name):
		"""Get the data needed for a projection of this camera into the VoxelCast env and respectively everything relevant to get the position of points on a cameras image in 3D (not cast, just on the oriented picture in 3D)

		:param name: Name of the camera
		:type name: str
		:return: vector to origin of camera, inverse transformation/rotation matrix, vector from origin to screen top left, vector from top left to top right corner of screen, vector from top left to bottom left of screen. All vectors are in the cameras coordinate system (camera facing down in -z with +y axis being "up" of the screen). 
		"""
		self.doesCamExist(name)

		cam = self.cameras[name]
		fovX = cam.data.angle_x # FOV in radians
		fovY = cam.data.angle_y
		#print(fovX, fovY)
		#if cam.data.lens_unit != "FOV":
		#	raise ValueError(f"PANIC! The camera {name} has a .data.lens_unit != 'FOV', which is not implemented. Please change to FOV :)")
		
		z = 4
		y = z*math.tan(fovY/2)
		x = z*math.tan(fovX/2)
		baseK = np.array([-x,y,-z]).transpose()
		yScreenK = np.array([0,-2*y,0]).transpose()
		xScreenK = np.array([2*x,0,0]).transpose()

		r, M = self.getCamPosOri(name)

		return r, M, baseK, xScreenK, yScreenK
