import cv2
import numpy as np

# Check if OpenCV was built with CUDA support
print("OpenCV CUDA Support:", cv2.cuda.getCudaEnabledDeviceCount() > 0)

# Print CUDA device information
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    device = cv2.cuda.Device()  # type: ignore[attr-defined]
    print("CUDA Device:", device.name())
    print("Compute Capability:", device.computeCapability())

# Print numpy version to ensure it's working properly
print("NumPy Version:", np.__version__)

