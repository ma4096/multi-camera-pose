# main file for multi-camera-pose (MCP)

import numpy as np

# this is a needed workaround, as importing bpy before creating a plt plot/object results in a segmentation fault when trying to plot.
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3
plt.plot([1])
plt.clf()
plt.close()
from timeit import default_timer as timer

from ultralytics import YOLO
import cv2
from VoxelCast import VoxelCast
from BlenderCam import BlenderCam
from LinearRays import LinearRays

class MCP:
	"""

	:param scene_file: Path to the blender scene where the position of cameras are defined.
	:type scene_file: str
	:param img: dictionary of camera names and where to get images. These can be paths to files or None (pass images yourself later on)
	:type img: dict<str>
	"""

	keypoints = ["nose", "left-eye", "right-eye", "left-ear", "right-ear", "left-shoulder", "right-shoulder", "left-elbow", "right-elbow", "left-wrist", "right-wrist", "left-hip", "right-hip", "left-knee", "right-knee", "left-ankle", "right-ankle"]

	def __init__(self, scene_file):
		b = BlenderCam(scene_file)
		self.camdat = b.camdat
		#self.cameras = img

		self.model = YOLO("yolo11n-pose.pt")
	
	def setCameras(self, img):
		self.cameras = img

	def infer(self, img):
		"""Get pose dict from an image with normalized coordinates of each point

		:param img: image to be inferred
		:type img: cv2-image object
		:returns: dict<keypoints(str) : [np.array([x,y])]> of normalized coordinates for each keypoint. If nothing was found, entry is None.
		"""
		results = self.model(img, show=True)[0] # not found a use yet to have it in a list? Maybe a point for the future... ####################################################
		#input("Press key to continue...")
		xyn = results.keypoints.xyn # list (multiple people) of lists (keypoints) of normalized coordinates
	
		d = {}
		for i in self.keypoints:
			d[i] = []
		
		for r in xyn: # each detected person
			for i,j in enumerate(r): # each keypoint
				#print(i,j)
				res = np.array(j) if (j[0] != 0 and j[1] != 0) else None
				d[self.keypoints[i]].append(res) 
		return d

	def castOld(self):
		"""This uses the old VoxelCast class, which has been superseded by LinearCast. 
		"""
		v = VoxelCast(([-7,8],[-7.5,7],[0,8]), 0.2, classes=self.keypoints) # these coordinates are just temporary for testing
		# TODO: automate or enable passing of coordinate ranges

		for cam in self.camdat.keys():
			print(f"Working on camera {cam}...")
			d = self.infer(self.cameras[cam])
			r, M, baseK, xScreenK, yScreenK = self.camdat[cam]
			print(f"Done with inferring, now casting rays...")
			for i,k in enumerate(self.keypoints):
				for point in d[k]:
					if type(point) == type(None):
						continue # keypoint was not found
					nu, eta = point[0], point[1]
					pointOnScreen = r + np.matmul(M, baseK + nu*xScreenK + eta*yScreenK)

					v.cast(r, pointOnScreen, typ=k)
			print("\n")
		print("Casting rays finished\n\n")
		v.showVoxels(onlyInter=False)
		#for k in self.keypoints:
		#	p = v.getInter(k)
		#	print(f"{k}: {p}")
		#v.showInter()
		return v

	def cast(self):
		"""Loads every image for each camera and uses YOLO to estimate the pose and casts a ray for each keypoint in the pose using LinearRays class.

		:return: LinearRays object
		"""
		lr = LinearRays(classes=self.keypoints)

		for cam in self.camdat:
			print(f"Working on camera {cam}...")
			d = self.infer(self.cameras[cam])
			r, M, baseK, xScreenK, yScreenK = self.camdat[cam]
			print(f"Done with inferring, now casting rays...")
			for i,k in enumerate(self.keypoints):
				for point in d[k]:
					if type(point) == type(None):
						continue # keypoint was not found
					nu, eta = point[0], point[1]
					pointOnScreen = r + np.matmul(M, baseK + nu*xScreenK + eta*yScreenK)

					lr.addRay(r, pointOnScreen, typ=k)
			print("\n")
		print("Casting rays finished\n\n")
		return lr

	def video(self):
		"""Generate a 3D pose sequence from multiple videos

		:param videos: name of cameras to path to videos dictionary. All videos must have the same length and start at the same time
		:type videos: dict<str>
		"""
		coords = {}
		for p in self.keypoints:
			coords[p] = []

		captures = {}
		last = None # tracks amount of frames
		fps = 0

		videos = self.cameras
		for c in videos:
			captures[c] = cv2.VideoCapture(videos[c])
			cur = int(captures[c].get(cv2.CAP_PROP_FRAME_COUNT))
			if last != None and cur != last:
				raise ValueError(f"Not all provided videos have the same amount of frames.")
			last = cur
			fps = captures[c].get(cv2.CAP_PROP_FPS)
		
		for i in range(last):
			frames = {}
			for c in videos:
				ret, frames[c] = captures[c].read()
			self.setCameras(frames)
			v = self.cast() # or just .cast()
			#v.showRays(classes=["left-hip","left-elbow"])
			inter = v.getAllInter(plot=True)
			for cl in self.keypoints:
				coords[cl].append(inter[cl])

		for c in captures:
			captures[c].release()

		return coords, fps, last

	def animateKeypoints(self, coords, fps, n_frames):
		"""Currently broken/under construction...
		"""
		flat = []
		for f in range(n_frames):
			flat.append([[],[],[]])
			for c in coords:
				#for i,f in enumerate(flat):
				for i in range(3):
					if not np.isnan(coords[c][f][i]):
						flat[f][i].append(float(coords[c][f][i]))
					else:
						flat[f][i].append(0)

		
		fig = plt.figure()
		ax = p3.Axes3D(fig)
		x,y,z = tuple(flat[0])
		print(x,y,z)
		points, = ax.plot(x,y,z, "*")
		txt = fig.suptitle("")

		def init():
			return ax,
		def update(num):#, x, y, z, points):
			txt.set_text(f"frame={num}")
			nx, ny, nz = tuple(flat[num])
			#print(nx,ny,nz)
			points.set_data(nx,ny)
			points.set_3d_properties(nz, "z")
			return points,# txt
		anim = FuncAnimation(fig, update, frames=list(range(n_frames)), init_func=init, blit=True) #fargs=(x,y,z, points))
		#anim.save("test.mp4")
		plt.show()

if __name__=="__main__":
	#i1 = "test_images/vidCamera1.mkv"
	#i2 = "test_images/vidCamera2.mkv"
	#i3 = "test_images/vidCamera3.mkv"
	i1 = cv2.imread("test_images/Camera1.png")
	i2 = cv2.imread("test_images/Camera2.png")
	i3 = cv2.imread("test_images/Camera3.png")
	c = {"Camera": i1, "Camera.001": i2, "Camera.002": i3}
	m = MCP("blender_test_scenes/threecams.blend")
	m.setCameras(c)
	#coords, fps, frames = m.video(c)
	#print(fps, frames)
	#with open("cache.txt","w") as file:
	#	file.write(str(coords))
	#m.animateKeypoints(coords,fps,frames)
	start = timer()
	lr = m.cast()
	inter = lr.getAllInter(plot=True)
	lr.showRays(classes=["left-eye", "left-hip"])
	#coords, fps, frames = m.video()
	#m.animateKeypoints(coords, fps, frames)
	end = timer()
	print(f"Time elapsed for casting and calculating intersections: {end-start} s")
	#lr.showRays(classes=["nose","right-hip"])