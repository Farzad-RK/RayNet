# import torch
# import torch.nn.functional as F
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from typing import Optional, Tuple, Dict
# import os
#
#
# class RayNetInference:
#     """
#     Inference class for eye-focused RayNet with ray casting capabilities
#     """
#
#     def __init__(self, model_path: str, device: str = 'cuda'):
#         self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
#
#         # Load model
#         self.model = self._load_model(model_path)
#         self.model.eval()
#
#     def _load_model(self, model_path: str):
#         """Load trained RayNet model"""
#         from raynet import RayNet
#         from backbone.repnext_utils import load_pretrained_repnext
#
#         # Load checkpoint
#         checkpoint = torch.load(model_path, map_location=self.device)
#         args = checkpoint.get('args', {})
#
#         # Initialize backbone
#         backbone_name = args.get('backbone_name', 'repnext_m3')
#         weight_path = args.get('weight_path', './repnext_m3_pretrained.pt')
#         backbone = load_pretrained_repnext(backbone_name, weight_path).to(self.device)
#
#         # Channel configuration
#         backbone_channels_dict = {
#             'repnext_m3': [64, 128, 256, 512],
#         }
#         in_channels_list = backbone_channels_dict[backbone_name]
#
#         # Initialize model
#         model = RayNet(
#             backbone=backbone,
#             in_channels_list=in_channels_list,
#             n_iris_landmarks=100,
#             panet_out_channels=256
#         ).to(self.device)
#
#         # Load weights
#         model.load_state_dict(checkpoint['model_state_dict'])
#
#         return model
#
#     def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = (448, 448)):
#         """
#         Preprocess image for model input
#         Args:
#             image: Input image (H, W, 3) in BGR format
#             target_size: Target size for model
#         Returns:
#             Preprocessed tensor
#         """
#         # Convert BGR to RGB
#         if len(image.shape) == 3 and image.shape[2] == 3:
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#         # Resize
#         image = cv2.resize(image, target_size)
#
#         # Normalize to [0, 1]
#         image = image.astype(np.float32) / 255.0
#
#         # Convert to tensor and add batch dimension
#         image_tensor = torch.from_numpy(image.transpose(2, 0, 1))
#         image_tensor = image_tensor.unsqueeze(0).to(self.device)
#
#         return image_tensor
#
#     @torch.no_grad()
#     def predict(self, image: np.ndarray):
#         """
#         Run inference on a single image
#         Args:
#             image: Input image
#         Returns:
#             Model outputs dictionary
#         """
#         # Preprocess
#         image_tensor = self.preprocess_image(image)
#
#         # Forward pass
#         outputs = self.model(image_tensor)
#
#         # Convert to numpy for easier handling
#         outputs_np = {}
#         for key, val in outputs.items():
#             if isinstance(val, torch.Tensor):
#                 outputs_np[key] = val.cpu().numpy()
#             elif isinstance(val, dict):
#                 outputs_np[key] = {}
#                 for k, v in val.items():
#                     if isinstance(v, torch.Tensor):
#                         outputs_np[key][k] = v.cpu().numpy()
#                     else:
#                         outputs_np[key][k] = v
#             else:
#                 outputs_np[key] = val
#
#         return outputs_np
#
#     def cast_gaze_to_screen(self,
#                             outputs: Dict,
#                             screen_distance: float = 600.0,
#                             screen_size: Tuple[float, float] = (520, 320),
#                             screen_center: Optional[np.ndarray] = None):
#         """
#         Cast gaze ray to screen plane
#         Args:
#             outputs: Model outputs
#             screen_distance: Distance to screen in mm
#             screen_size: Screen size (width, height) in mm
#             screen_center: Screen center position (default: straight ahead)
#         Returns:
#             Dictionary with screen intersection information
#         """
#         # Get ray parameters
#         ray_origin = outputs['ray_origin'][0]  # Remove batch dimension
#         ray_direction = outputs['ray_direction'][0]
#
#         # Define screen plane
#         if screen_center is None:
#             screen_center = np.array([0, 0, screen_distance])
#
#         screen_normal = np.array([0, 0, -1])  # Screen facing the user
#
#         # Compute intersection
#         # t = dot(normal, center - origin) / dot(normal, direction)
#         t = np.dot(screen_normal, screen_center - ray_origin) / (np.dot(screen_normal, ray_direction) + 1e-8)
#
#         # Intersection point in 3D
#         intersection_3d = ray_origin + t * ray_direction
#
#         # Convert to screen coordinates
#         screen_x = intersection_3d[0] - screen_center[0]
#         screen_y = intersection_3d[1] - screen_center[1]
#
#         # Convert to pixel coordinates (assuming screen resolution)
#         screen_width_mm, screen_height_mm = screen_size
#         pixel_x = (screen_x / screen_width_mm + 0.5) * 1920  # Assuming 1920x1080 resolution
#         pixel_y = (screen_y / screen_height_mm + 0.5) * 1080
#
#         return {
#             'intersection_3d': intersection_3d,
#             'screen_coords_mm': (screen_x, screen_y),
#             'pixel_coords': (int(pixel_x), int(pixel_y)),
#             'distance': t,
#             'on_screen': abs(screen_x) <= screen_width_mm / 2 and abs(screen_y) <= screen_height_mm / 2
#         }
#
#     def visualize_3d_reconstruction(self, outputs: Dict, save_path: Optional[str] = None):
#         """
#         Create 3D visualization of eye reconstruction and gaze
#         """
#         fig = plt.figure(figsize=(20, 10))
#
#         # Plot 1: Full 3D scene with both eyes
#         ax1 = fig.add_subplot(121, projection='3d')
#
#         # Plot eyeballs
#         for eye in ['left', 'right']:
#             eyeball = outputs[f'eyeball_{eye}']
#             center = eyeball['center'][0]
#             radius = eyeball['radius'][0]
#
#             # Create sphere
#             u = np.linspace(0, 2 * np.pi, 30)
#             v = np.linspace(0, np.pi, 20)
#             x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
#             y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
#             z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
#
#             ax1.plot_surface(x, y, z, alpha=0.3, color='pink' if eye == 'left' else 'lightblue')
#
#             # Plot iris landmarks
#             iris = outputs['iris_landmarks'][eye][0]
#             ax1.scatter(iris[:, 0], iris[:, 1], iris[:, 2],
#                         c='blue' if eye == 'left' else 'green', s=2)
#
#             # Plot pupil center
#             pupil = outputs['pupil_centers'][eye][0]
#             ax1.scatter(pupil[0], pupil[1], pupil[2],
#                         c='red', s=50, marker='*')
#
#             # Plot optical and visual axes
#             origin = eyeball['center'][0]
#             optical = outputs[f'optical_axis_{eye}'][0]
#             visual = outputs[f'visual_axis_{eye}'][0]
#
#             ax1.quiver(origin[0], origin[1], origin[2],
#                        optical[0] * 30, optical[1] * 30, optical[2] * 30,
#                        color='orange', alpha=0.5, arrow_length_ratio=0.1)
#             ax1.quiver(origin[0], origin[1], origin[2],
#                        visual[0] * 30, visual[1] * 30, visual[2] * 30,
#                        color='red', alpha=0.7, arrow_length_ratio=0.1)
#
#         # Plot combined gaze ray
#         ray_origin = outputs['ray_origin'][0]
#         ray_direction = outputs['ray_direction'][0]
#         gaze_point = outputs['gaze_point_3d'][0]
#
#         # Draw ray
#         t = np.linspace(0, np.linalg.norm(gaze_point - ray_origin), 100)
#         ray_points = ray_origin[:, np.newaxis] + ray_direction[:, np.newaxis] * t
#         ax1.plot(ray_points[0], ray_points[1], ray_points[2],
#                  'r-', linewidth=3, label='Gaze Ray')
#
#         # Mark gaze point
#         ax1.scatter(gaze_point[0], gaze_point[1], gaze_point[2],
#                     c='green', s=100, marker='o', label='Gaze Point')
#
#         ax1.set_xlabel('X (mm)')
#         ax1.set_ylabel('Y (mm)')
#         ax1.set_zlabel('Z (mm)')
#         ax1.set_title('3D Eye Reconstruction and Gaze')
#         ax1.legend()
#
#         # Set equal aspect ratio
#         max_range = 50
#         ax1.set_xlim([-max_range, max_range])
#         ax1.set_ylim([-max_range, max_range])
#         ax1.set_zlim([-max_range, max_range])
#
#         # Plot 2: Top-down view with screen intersection
#         ax2 = fig.add_subplot(122)
#
#         # Draw eyes from above
#         for eye in ['left', 'right']:
#             center = outputs[f'eyeball_{eye}']['center'][0]
#             ax2.scatter(center[0], center[2], s=200,
#                         c='pink' if eye == 'left' else 'lightblue',
#                         label=f'{eye.capitalize()} Eye')
#
#         # Draw gaze ray
#         ray_x = [ray_origin[0], gaze_point[0]]
#         ray_z = [ray_origin[2], gaze_point[2]]
#         ax2.plot(ray_x, ray_z, 'r-', linewidth=2, label='Gaze Ray')
#
#         # Draw screen
#         screen_z = 600  # mm from eyes
#         screen_width = 520  # mm
#         ax2.plot([-screen_width / 2, screen_width / 2], [screen_z, screen_z],
#                  'k-', linewidth=3, label='Screen')
#
#         # Mark intersection
#         ax2.scatter(gaze_point[0], gaze_point[2], c='green', s=100,
#                     marker='*', label='Gaze Point', zorder=5)
#
#         ax2.set_xlabel('X (mm)')
#         ax2.set_ylabel('Z (mm)')
#         ax2.set_title('Top-Down View')
#         ax2.legend()
#         ax2.grid(True, alpha=0.3)
#         ax2.set_aspect('equal')
#
#         plt.tight_layout()
#
#         if save_path:
#             plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         else:
#             plt.show()
#
#         plt.close()
#
#     def process_video(self, video_path: str, output_path: str,
#                       screen_distance: float = 600.0):
#         """
#         Process video and track gaze on screen
#         """
#         cap = cv2.VideoCapture(video_path)
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#         # Video writer
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#
#         gaze_history = []
#
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             # Run inference
#             outputs = self.predict(frame)
#
#             # Cast to screen
#             screen_result = self.cast_gaze_to_screen(outputs, screen_distance)
#
#             # Store gaze point
#             gaze_history.append(screen_result['pixel_coords'])
#
#             # Draw on frame
#             if screen_result['on_screen']:
#                 # Draw gaze point
#                 cv2.circle(frame, screen_result['pixel_coords'], 20, (0, 255, 0), -1)
#
#                 # Draw gaze trail
#                 if len(gaze_history) > 1:
#                     for i in range(1, min(len(gaze_history), 10)):
#                         cv2.line(frame, gaze_history[-i], gaze_history[-i - 1],
#                                  (0, 255, 0), 2)
#
#             # Add text info
#             cv2.putText(frame, f"Gaze: {screen_result['pixel_coords']}",
#                         (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#             cv2.putText(frame, f"Distance: {screen_result['distance']:.1f}mm",
#                         (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#
#             out.write(frame)
#
#         cap.release()
#         out.release()
#         cv2.destroyAllWindows()
#
#         return gaze_history
#
#
# # Example usage
# if __name__ == "__main__":
#     # Initialize inference
#     model_path = "checkpoints_eye/raynet_eye_epoch50.pth"
#     inferencer = RayNetInference(model_path)
#
#     # Process single image
#     image = cv2.imread("test_image.jpg")
#     outputs = inferencer.predict(image)
#
#     # Cast gaze to screen
#     screen_result = inferencer.cast_gaze_to_screen(outputs)
#     print(f"Gaze point on screen: {screen_result['pixel_coords']}")
#     print(f"Distance to gaze point: {screen_result['distance']:.1f}mm")
#     print(f"On screen: {screen_result['on_screen']}")
#
#     # Visualize 3D reconstruction
#     inferencer.visualize_3d_reconstruction(outputs, save_path="reconstruction.png")
#
#     # Process video (if available)
#     # gaze_history = inferencer.process_video("input_video.mp4", "output_with_gaze.mp4")