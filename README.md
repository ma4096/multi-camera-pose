# multi camera pose
Using multiple cameras for pose estimation in 3D.

This project utilizes Ultralytics YOLO11 for pose estimation on pictures taken from multiple angles. Reconstructing the camera position and data from a blender scene, rays are cast from each cameras origin through the point on its oriented image through a 3D space. Rays are separated by the type of point they represent (hand, eye, ...). The weighted average of the center of the shortest vector between each ray of a certain class is determined to be that points position in 3D.

Heavily inspired by [Consistently Inconsistent's video](https://www.youtube.com/watch?v=m-b51C82-UE) (last accessed 4/13/25).

## Current workarounds
Importing bpy (blender python API) as is done in `BlenderCam.py` before creating a plt figure results in a "segmentation fault (core dumped)". Head every use of `BlenderCam.py` with this:

```python
import matplotlib.pyplot as plt
plt.plot([1])
plt.clf() # clear the figure
plt.close() # close the window of the figure (otherwise there will be an additional empty window next time you plt.show())
"""You can put more code between """
from BlenderCam import BlenderCam
```

## Usage
`MCP.py` implements the full workflow, including pose estimation and casting rays. See that files last lines.

## Credits
The 3D model used for examples is from [Real World Textured Things](https://texturedmesh.isti.cnr.it/) "Arnau 3D scan" (nr. 178)