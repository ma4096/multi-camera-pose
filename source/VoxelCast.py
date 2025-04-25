import numpy as np
import math
import matplotlib.pyplot as plt
#import warnings # not used right now

class VoxelCast:
	"""Initialize a voxel based coordinate system and cast rays based on two points through it, detecting collisions. THIS IS CURRENTLY NOT IN USE, as it has been replaced by LinearRays.py

	:param ranges: ranges of axes coordinates in the order x,y,z like ([-1,1],[0,2],[2,6]) in an arbitrary length unit. Each second entry must be larger than the first.
	:type ranges: tuple<np.array>
	:param length: length of a side of a voxel in length unit
	:type length: float
	:param classes: list of types of rays. Rays of different types are not able to collide/intersect.
	:type classes: optional list<str>
	"""
	rays = {} # ray counter
	intersections = {} # storage for interceptions
	#def __init__(self, dx,dy,dz, length, classes=["default"]):
	def __init__(self, ranges, length, classes=["default"]):
		#self.x, self.y, self.z = np.indices((dx,dy,dz))
		self.xr,self.yr,self.zr = ranges
		self.ranges = ranges
		shape = tuple([int((i[1]-i[0])/length) for i in ranges])
		self.lower_corner = np.array([i[0] for i in ranges]).transpose()
		self.upper_corner = np.array([i[1] for i in ranges]).transpose()
		
		print(shape)
		#print(self.lower_corner, self.upper_corner)

		self.c = {}
		self.classes = classes
		d = {}
		for c in classes:
			self.c[c] = np.zeros(shape, dtype=object)
			self.rays[c] = 0
					
		self.length = length

	def TypIsInClasses(self, typ):
		if typ not in self.classes:
			raise ValueError(f"The provided type {typ} of the array was not initiated when creating the VoxelCast Object. Initiated were: {self.classes}")

	# TODO: allow for negative coordinates -> affine not proportional
	def getPFI(self, p):
		"""Get place from index, lowest corner of voxel
		:param p: indexes of coordinate system
		:type p: np.array
		:returns: Coordinates in length units
		"""
		return p*self.length + self.lower_corner

	def getIFP(self, c):
		"""Reverse of getPFI, see that. Rounds down to get coordinates of according voxel
		"""
		return ((c - self.lower_corner)/self.length).astype(int)

	def cast(self,p1,p2,remove=False, typ="default"):
		"""Draw a cast starting at p1 through p2 until the end of the coordinate system

		:param p1,p2: Two points in 3D laying inside the defined coordinate system in length units (not indexes).
		:type p1,p2: np.array<float>
		:param remove: Remove a ray (subtracts 1 rather than add 1).
		:type remove: optional bool
		:param typ: Type of ray, e.g. "hand" for pose estimation
		:type typ: optional str
		"""
		self.TypIsInClasses(typ) # otherwise raise exception
			
		# remove or not to remove
		change = -1 if remove else 1

		# determine the biggest derivative of the ray
		db = p2-p1 # each element is the derivative of b = p2*t+p1*(1-t)
		t = 0
		step = self.length/max(abs(db)) /2 # testwise smaller step
		i = 0 # track the amount of iterations for logging
		last = np.nan # store the last checked voxel to prevent counting double
		while True:
			b = p2*t + p1*(1-t)
			bc = self.getIFP(b)

			if np.array_equal(bc, last): # check if the last voxel updated is the same as now, then skip this one. Otherwise update the last voxel's coordinates.
				#print(f"Skipped {bc} at iteration {i}")
				t += step
				continue
			elif np.count_nonzero(abs(bc-last) > 1) > 0: # if between the current location and the last, any dimension increased by more than 1, voxel was skipped in the ray. This goes back half a step and tries again.
				print(f"Found a skipped coordinate from {last} to {bc}, returning a bit.")
				t -= step/2
				continue
			if np.any(bc < 0): # as python allows for negative indices
				break
			last = bc

			try:
				tbc = tuple(bc)
				self.c[typ][tbc] += change # -1 if remove, 1 else
				if self.c[typ][tbc] < 0:
					self.c[typ][tbc] = 0

				# TODO: implement padding around the ray
			except Exception as e: # array index out of bounds
				#print(e)
				break
			t += step
			i += 1
		print(f"It took {i} evaluations to cast a ray from {p1} through {p2}.")
		self.rays[typ] += 1
			
	def showVoxels(self, onlyInter=False):
		"""Shows a plot of the current coordinate system in 3D
		"""
		colorsAvailable = ['b','g','r','c','m','y']
		
		ax = plt.figure().add_subplot(projection="3d")
		for j,t in enumerate(self.classes):
			colors = np.empty(self.c[t].shape, dtype="object")
			for i in range(1,self.rays[t]+1):
				colors[self.c[t] == i] = colorsAvailable[(i-1)%6]
				if i > 1:
					print(f"Amount of {i} for type {t}: {colors[self.c[t] == i].sum()}")

			toPlot = None
			if onlyInter:
				toPlot = self.c[t] > 1
			else:
				toPlot = self.c[t].astype(bool)
					
			ax.voxels(toPlot, facecolors=colorsAvailable[j%6], edgecolors=colors)
		
		ax.set_aspect("equal","box")
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.set_zlabel("z")
		x0, y0, z0 = tuple(self.getIFP(self.upper_corner))
		x1, y1, z1 = tuple(self.getIFP(self.lower_corner))
		ax.scatter([x0,x1],[y0,y1],[z0,z1])
		plt.show()
		
		#plt.savefig("test.png")

	def getInter(self, typ="default", asLengthUnit=True, noReduce=False):
		"""Returns the positions of every intersection (indices) and their level (amount of rays intersecting) for a specified type of ray
		:param typ: Amount of rays intersecting
		:type typ: str
		:returns: dict of level (str) : list of np.arrays<float> with coordinates (indices) of intersections.
		"""
		self.TypIsInClasses(typ)
		# TODO: when implementing padding -> get middle of cluster
		d = {}
		arr = self.c[typ]
		tot = np.array([0,0,0])
		div = 0
		for i in range(2,arr.max()+1): # start at 2 because 1 is not an intersection
			at = arr == i
			lol = np.transpose(np.where(at))
			#if not noReduce:
			#	m = np.array([0,0,0])
			#	for inter in lol: 
			#		m += np.array(inter)
			#	m = m/len(lol)
			#	lol = [list(m)]

			n = [np.array(m) for m in lol]
			if asLengthUnit:
				n = [self.getPFI(np.array(m)) for m in lol]
			d[str(i)] = n

			for m in n:
				tot = tot + i*m
				div += i
		
		if d == {}:
			print(f"No intersections found for type {typ}.")
			#warnings.warn(f"No intersections found for type {typ}.")
		else:
			tot = tot/div
			d["total"] = tot

		self.intersections[typ] = tot
		return d

	def getAllInter(self):
		for c in self.classes:
			self.getInter(typ=c)
		return self.intersections

	def getRayTypes(self):
		return self.rays

	def showInter(self):
		#fig = figure()
		ax = plt.figure().add_subplot(projection="3d")
		intersections = {}
		for c in self.classes:
			intersections[c] = self.getInter(c)
			if intersections[c]: # not empty
				coords = intersections[c]["total"] # currently there are no more than level 2
				print(f"COORDS of {c}: {coords}")
				ax.scatter(*coords)
				ax.text(*coords, c)
		ax.set_aspect("equal","box")
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')
		plt.show()