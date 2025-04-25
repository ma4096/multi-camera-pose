import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
colors_list = list(colors._colors_full_map.values())

class LinearRays():
	"""More reliable solver for not-quite intersecting rays using linear algebra instead of voxels

	"""

	def __init__(self, classes="default"): # no ranges are needed (yet)
		self.rays = {}
		self.classes = classes
		for c in classes:
			self.rays[c] = []

	def addRay(self, p1, p2, typ="default"):
		"""Add a ray to its classes environment
		
		:param p1,p2: Two points on the ray, vertical arrays!
		:type p1,p2: np.array
		"""
		dif = p2-p1
		self.rays[typ].append([p1, dif])
	
	def showRays(self, length=4, classes=["default"]): # for all classes
		ax = plt.figure().add_subplot(projection="3d")
		colors_list = list(colors._colors_full_map.values())
		for i,c in enumerate(classes):
			#coords = [[],[],[]]
			for ray in self.rays[c]:
				r, dif = tuple(ray)
				start = r
				middle = r + length/2*dif
				end = r + length*dif
				co = np.array([start, middle, end]).transpose()
				ax.plot(*co, color=colors_list[i])
		ax.set_aspect("equal","box")
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')
		plt.show()

	def getInter(self, typ="default"):
		"""Get the weighted/averaged intersection estimate for all rays of a certain class. Computes in O(n^2) for n rays.

		:param typ: class of the rays
		:type typ: str
		"""
		if typ not in self.classes:
			raise ValueError(f"The class {typ} has not been initialized when setting up the LinearRays object. Initialized where {self.classes}")

		rays = self.rays[typ]
		num = np.array([0,0,0]) # numerator for maximum
		den = 0
		for i,g in enumerate(rays):
			p = g[0]
			b = g[1]
			for j,h in enumerate(rays[i:]):
				q = h[0]
				c = h[1]

				if (b==c).all(): # if the rays are parallel, skip it
					continue

				z = p-q
				# check if the rays really intersect
				a = z[:2] # take the first 2 elements
				A = np.array([[-b[0], c[0]],[-b[1], c[1]]])
				lamb, phi = tuple(np.linalg.solve(A,a))
				if z[2] == -b[2]*lamb + c[2]*phi:
					m = p + lamb*b
					w = 1e5 # just any high value
					print(f"Hooray! A real intersection at {m} was found! This is super rare when working with this kind of position :)")

				# else find the middle of the minimum distance vector
				else:
					cb = np.dot(c,b)
					bb = np.dot(b,b)
					cc = np.dot(c,c)
					bpq = np.dot(b,z)
					cpq = np.dot(c,z)
					a = np.array([[-bpq],[-cpq]])
					A = np.array([[bb, -cb],[cb, -cc]])

					lamb, phi = tuple(np.linalg.solve(A,a))

					m = (p+q+lamb*b+phi*c)/2

					n = np.cross(b,c)
					w = np.linalg.norm(n)/np.linalg.norm(np.dot(z,n))

				num = num + w*m
				den += w
		
		I = num/den # weighted intersection average
		# throws a warning when dividing by zero -> i can handle nan myself :)
		return I

	def getAllInter(self, plot=False):
		aI = {}
		ax = 0
		if plot:
			ax = plt.figure().add_subplot(projection="3d")

		for c in self.classes:
			aI[c] = self.getInter(typ=c)

			if plot:
				ax.scatter(*aI[c])
				ax.text(*aI[c], c)

		if plot:
			ax.set_aspect("equal","box")
			ax.set_xlabel('x')
			ax.set_ylabel('y')
			ax.set_zlabel('z')
			plt.show()
		return aI
