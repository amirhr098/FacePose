# 6D
import math
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import sixutils
from PIL import Image

# Synergy
from utils.ddfa import ToTensor, Normalize
from model_building import SynergyNet
from utils.inference import crop_img, predict_sparseVert, predict_pose
cudnn.benchmark = True
cudnn.enabled = True
from FaceBoxes import FaceBoxes

face_boxes = FaceBoxes()
IMG_SIZE = 120
GPU = 0

tdx_buf = []
tdy_buf = []
rots_buf = []
counter = 0

six_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(
                                              224), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

syn_transforms = transforms.Compose([ToTensor(), Normalize(mean=127.5, std=128)])


class SixDRepNet360(nn.Module):
    def __init__(self, block, layers, fc_layers=1):
        self.inplanes = 64
        super(SixDRepNet360, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)

        self.linear_reg = nn.Linear(512*block.expansion,6)
      
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.linear_reg(x)        
        out = sixutils.compute_rotation_matrix_from_ortho6d(x)

        return out


def load_synModel(args, checkpoint_fp='pretrained/best.pth.tar'):
    """
    Load the pre-trained SynergyNet model.

    Args:
        args: Arguments object containing the model parameters.

    Returns:
        syn_model (nn.Module): The loaded SynergyNet model.
    """
    # Load pre-trained syn_model
    checkpoint_fp = checkpoint_fp  # File path of the checkpoint
    args.arch = 'mobilenet_v2'  # Model architecture
    args.devices_id = [0]  # Device ID

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    
    # Create the SynergyNet model
    syn_model = SynergyNet(args)
    syn_model_dict = syn_model.state_dict()

    # Remove the 'module.' prefix from the keys of the checkpoint since the model was trained using multiple GPUs
    for k in checkpoint.keys():
        syn_model_dict[k.replace('module.', '')] = checkpoint[k]

    # Load the checkpoint into the model
    syn_model.load_state_dict(syn_model_dict, strict=False)

    # Move the model to GPU and set it to evaluation mode
    syn_model = syn_model.cuda()
    syn_model.eval()

    return syn_model


def get_syn(syn_model, img_ori):
    """
    Get the 6D facial pose and 2D facial landmarks of a face in an image.

    Args:
        syn_model (nn.Module): The SynergyNet model for facial pose and landmark prediction.
        img_ori (np.ndarray): The input image.

    Returns:
        angles (list): The 3D facial pose angles.
        tdx (float): The x-coordinate of the 30th facial landmark.
        tdy (float): The y-coordinate of the 30th facial landmark.
    """
    # Preparation
    # Get the bounding box of the face in the image
    rect = face_boxes(img_ori)[0]
    roi_box = rect

    # Enlarge the bounding box a little and do a square crop
    HCenter = (rect[1] + rect[3]) / 2
    WCenter = (rect[0] + rect[2]) / 2
    side_len = roi_box[3] - roi_box[1]
    margin = side_len * 1.2 // 2
    roi_box[0], roi_box[1], roi_box[2], roi_box[3] = WCenter - margin, HCenter - margin, WCenter + margin, HCenter + margin

    # Crop and Transform to Tensor
    img = crop_img(img_ori, roi_box)
    img = cv2.resize(img, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    input = syn_transforms(img).unsqueeze(0)

    # Predicting
    with torch.no_grad():
        input = input.cuda()
        param = syn_model.forward_test(input)
        param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

        # Inferences
        # Predict facial landmarks
        lmks = predict_sparseVert(param, roi_box, transform=True)
        # Predict 3D facial pose
        angles, translation = predict_pose(param, roi_box)

    # Get the 30th facial landmark coordinates
    tdx = lmks[0, 30]
    tdy = lmks[1, 30]

    return angles, tdx, tdy

def load_sixModel(six_path = 'pretrained/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth'):
    """
    Load the SixDRepNet360 model for 6D facial pose estimation.

    Returns:
        model (torch.nn.Module): The loaded SixDRepNet360 model.
    """
    # Define the path to the pre-trained model checkpoint
    snapshot_path = six_path
    
    # Instantiate the SixDRepNet360 model
    model = SixDRepNet360(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 6)
    
    # Load the pre-trained model weights
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)  
    
    # Move the model to the GPU and set it to evaluation mode
    model.cuda(GPU)
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    
    return model

def get_rt(six_model, syn_model, img_ori, mode_num, frame_weights = [0.5, 0.5]):
    """
    Get the rotation and translation values from the given image using the SixDRepNet360 model and the SynergyNet model.

    Args:
        six_model (torch.nn.Module): The SixDRepNet360 model for 6D facial pose estimation.
        syn_model (torch.nn.Module): The SynergyNet model for facial landmark detection.
        img_ori (numpy.ndarray): The input image.
        mode_num (int): The mode number to determine when to update the rotation values. If -1, only SynergyNet Model is used.

    Returns:
        List[float]: The rotation values in degrees.
        List[float]: The translation values.
    """
    global tdx_buf, tdy_buf, rots_buf, counter
    buffer_num = len(frame_weights)
    # Transform image to Tensor
    # Predicting
    with torch.no_grad():
        # Predict facial angles and translation values using the SynergyNet model
        angles, tdx, tdy = get_syn(syn_model, img_ori)

        # If the counter is a multiple of the mode number and mode number is not -1, update the rotation values
        if mode_num != -1 and counter % mode_num == 0:
            image = six_transforms(Image.fromarray(img_ori))
            image = image.unsqueeze(0).cuda(GPU)
            R_pred = six_model(image)
            euler = sixutils.compute_euler_angles_from_rotation_matrices(
                R_pred) * 180 / np.pi
            angles[1] = (euler[:, 0].cpu().numpy()[0] + angles[1]) / 2
            angles[0] = (euler[:, 1].cpu().numpy()[0] + angles[0]) / 2
            angles[2] = (euler[:, 2].cpu().numpy()[0] + angles[2]) / 2
            counter = 0

        counter += 1

        # If the translation values have been set, calculate the averaged values
        if len(tdx_buf) == (buffer_num-1):
            tdx = (tdx * frame_weights[0] + np.sum(np.array(tdx_buf) * np.array(frame_weights[1:]))) / sum(frame_weights)
            tdy = (tdy * frame_weights[0] + np.sum(np.array(tdy_buf) * np.array(frame_weights[1:]))) / sum(frame_weights)
            angles[0],angles[1],angles[2] = (np.array([angles[0],angles[1],angles[2]]) * frame_weights[0] + np.sum((np.array(rots_buf).T * np.array(frame_weights[1:])).T, axis=0)) / sum(frame_weights)
            # tdx = (tdx + tdx_buf) / 2
            # tdy = (tdy + tdy_buf) / 2
            # angles[0] = (angles[0] + rots_buf[0]) / 2
            # angles[1] = (angles[1] + rots_buf[1]) / 2
            # angles[2] = (angles[2] + rots_buf[2]) / 2

        tdx_buf.insert(0,tdx)
        tdy_buf.insert(0,tdy)
        rots_buf.insert(0,[angles[0],angles[1],angles[2]])
        if(len(tdx_buf) >= buffer_num):
            del tdx_buf[-1]
            del tdy_buf[-1]
            del rots_buf[-1]

    return [angles[0], angles[1], angles[2]], [int(tdx), int(tdy)]
    


class FacePose:
    def __init__(self, synergymodel_path = 'pretrained/best.pth.tar', sixdrepnetmodel_path = 'pretrained/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth', frame_weights = None):
        """
        Initialize the FacePose object with the given paths.

        Args:
            synergymodel_path (str): Path to the SynergyNet model checkpoint.
            sixdrepnetmodel_path (str): Path to the 6DRepNet model checkpoint.
            frame_weights (list, optional): The weights of the past frames and current frame for pose calculation. The order of input is as follow: [cuurent_frame , first_past_frame, second_past_frame, ...]
        """
        # Parse command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--png", action="store_true", help="if images are with .png extension")
        parser.add_argument('--img_size', default=360, type=int)
        parser.add_argument('-b', '--batch-size', default=1, type=int)
        args = parser.parse_args()

        # Load the SynergyNet model
        self.syn_model = load_synModel(args,checkpoint_fp=synergymodel_path)  # Load the SynergyNet model

        # Load the 6DRepNet model
        self.six_model = load_sixModel(six_path=sixdrepnetmodel_path)  # Load the 6DRepNet model

        self.default_frame_weights = frame_weights
        

    def get_pose(self, img, mode=1, frame_weights=[0.5, 0.5]):
        """
        Get the pose of a face from an image using the SynergyNet and 6DRepNet models.

        Args:
            img (numpy.ndarray): The image containing the face.
            mode (int, optional): The mode to use. Defaults to 1.

                - 1: Uses both models in the same time.
                - 2: Uses only the Synergy Model.
                - 3: Uses both models in rotating manner.

            frame_weights (list, optional): The weights of the past frames and current frame for pose calculation. The order of input is as follow: [cuurent_frame , first_past_frame, second_past_frame, ...]

        Returns:
            list: A list containing the rotation and nose coordinates [Yaw, Pitch, Roll], [Nose X, Nose Y].
        """
        if self.default_frame_weights is not None:
            frame_weights = self.default_frame_weights

        # Smoothing Filter (Good for High-Res Images)
        if (img.shape[0] > 360) or (img.shape[1] > 360):
            kernel = np.ones((5,5),np.float32)/25
            img = cv2.filter2D(img,-1,kernel)

        if mode == 1:
            # This Mode uses both models in the same time
            return get_rt(self.six_model, self.syn_model, img, mode_num=1, frame_weights=frame_weights)
        elif mode == 2:
            # This Mode uses only the Synergy Model
            return get_rt(self.six_model, self.syn_model, img, mode_num=-1, frame_weights=frame_weights)
        elif mode == 3:
            # This Mode uses both models in rotating manner
            # (one 6DRepNet and one SynergyNet)
            return get_rt(self.six_model, self.syn_model, img, mode_num=2, frame_weights=frame_weights)
        else:
            # This Mode uses only the Synergy Model
            return get_rt(self.six_model, self.syn_model, img, mode_num=-1, frame_weights=frame_weights)
        # Output is [Yaw, Pitch, Roll], [Nose X, Nose Y]
    
    def __call__(self, img, mode=1, frame_weights=[0.5, 0.5]):
        """
        Get the pose of a face from an image using the SynergyNet and 6DRepNet models.

        Args:
            img (numpy.ndarray): The image containing the face.
            mode (int, optional): The mode to use. Defaults to 1.

                - 1: Uses both models in the same time.
                - 2: Uses only the Synergy Model.
                - 3: Uses both models in rotating manner.

        Returns:
            list: A list containing the rotation and nose coordinates [Yaw, Pitch, Roll], [Nose X, Nose Y].
        """
        # Obtain the pose of the face from the image
        return self.get_pose(img, mode, frame_weights)

    def __del__(self):
        """
        Deletes the objects `self.syn_model` and `self.six_model`.

        This method is called when the object is about to be destroyed.
        It ensures that the memory used by these objects is freed.
        """
        
        # Delete the SynergyNet model
        del self.syn_model
        
        # Delete the 6DRepNet model
        del self.six_model

    def reset_buffers(self):
        """
        Resets the buffers used for tracking.

        This function resets the `tdx_buf`, `tdy_buf`, and `rots_buf` buffers to their initial values.
        It also resets the `counter` to 0.

        """
        # Reset the translation buffer x
        tdx_buf = []

        # Reset the translation buffer y
        tdy_buf = []

        # Reset the rotation buffer
        rots_buf = []

        # Reset the counter to 0
        counter = 0

    def video_pose(self, cam: cv2.VideoCapture, mode=1, frame_num_max=np.inf, show=False, frame_weights=[0.5, 0.5]):
        """
        Obtains the pose of a face from each frame of a video stream.

        Args:
            cam (cv2.VideoCapture): The video capture object.
            mode (int, optional): The mode to use. Defaults to 1.

                - 1: Uses both models in the same time.
                - 2: Uses only the Synergy Model.
                - 3: Uses both models in rotating manner.

            frame_num_max (int, optional): The maximum number of frames to process. Defaults to np.inf.
            show (Boolean, optional): If True, an OpenCV window will show the results of each frame.
            frame_weights (list, optional): The weights of the past frames and current frame for pose calculation. The order of input is as follow: [cuurent_frame , first_past_frame, second_past_frame, ...]

        Returns:
            numpy.ndarray: An array containing the rotation and nose coordinates [Yaw, Pitch, Roll], [Nose X, Nose Y] for each frame.
        """

        if self.default_frame_weights is not None:
            frame_weights = self.default_frame_weights

        # Check if the video capture object is open
        try:
            if not cam.isOpened():
                print("Error opening video stream")
                return None
        except Exception as e:
            print(f"Error opening video stream. `cam` most be a OpenCV VideoCapture object. {e}")
            return None

        # Initialize an empty list to store the poses for each frame
        all_poses = []

        # Loop through each frame of the video
        while True:
            # Read the next frame from the video
            ret, frame = cam.read()

            # If no more frames are left, exit the loop
            if (not ret) or (len(all_poses) >= frame_num_max):
                break

            # Obtain the pose of the face from the image
            pose = self.get_pose(frame, mode, frame_weights=frame_weights)

            # Append the pose to the list
            all_poses.append(pose)

            # if show is True, show the output on the images with OpenCV

            # You should change the `cv2.resize` line as needed. 
            # Also `size` arg in `draw axis` if it's too small or big
            if show:
                frame = sixutils.draw_axis(frame,pose[0][0],pose[0][1],pose[0][2],pose[1][0],pose[1][1], size = 200)
                frame = cv2.resize(frame,(400,600))
                cv2.imshow("Pose", frame)
                cv2.waitKey(1)
        
        if show:
            cv2.destroyAllWindows()
        
        # Return the array of poses
        return np.array(all_poses, dtype=object)
            