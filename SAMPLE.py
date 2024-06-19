from FacePose import FacePose
import cv2

# Create an instance of the FacePose class
print("Loading FacePose Model")
facepose = FacePose(
    synergymodel_path='pretrained/best.pth.tar', 
    sixdrepnetmodel_path='pretrained/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth')

# Usage Examples
# Sample 1
# Load an image
image = cv2.imread("test.jpg")
# Get the pose from a image

# Calling facepose itself
rotation, translation = facepose(image, mode=3)

# Calling the get_pose function
rotation, translation = facepose.get_pose(image, mode=3)

print("Rotation [Yaw, Pitch, Roll]")
print(rotation)
print("Translation [X, Y] (Nosetip X and Y)")
print(translation)

# Sample 2
# Get the pose from a video
cam = cv2.VideoCapture("test.mp4")
all_poses = facepose.video_pose(cam, mode=3, frame_num_max=10, show=True, frame_weights=[10,3,3])

print("Total number of frames: ", len(all_poses))
print(len(all_poses))

print("First frame Rotation [Yaw, Pitch, Roll]")
print(all_poses[0][0])
print("First frame Translation [X, Y] (Nosetip X and Y)")
print(all_poses[0][1])

# Other Functions
facepose.reset_buffers()

del facepose
