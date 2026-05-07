import numpy as np
import cv2
import os
import pickle
import matplotlib.pyplot as plt


def DrawArrow(img, center, direction, length, ins_matrix, color, thickness):
    begin = np.dot(ins_matrix, center.reshape(3, 1))
    begin /= begin[2, 0]
    begin = begin[:2, 0]
    end = center + length * direction
    end = np.dot(ins_matrix, end.reshape(3, 1))
    end /= end[2, 0]
    end = end[:2, 0]
    cv2.arrowedLine(img, (round(begin[0]), round(begin[1])), (round(end[0]), round(end[1])), color, thickness)

def Draw_X(img, centers, length, color, width):
    length = round(length)
    if len(centers.shape) == 1:
        centers = centers.reshape(1, -1)
    for center in centers:
        center = (round(center[0]), round(center[1]))
        img = cv2.line(img, (center[0] - length, center[1] - length), (center[0] + length, center[1] + length), color, width)
        img = cv2.line(img, (center[0] - length, center[1] + length), (center[0] + length, center[1] - length), color, width)
    return img

def Draw2DPts(img, pts, color=(0, 255, 0), radius=1):
    if len(pts.shape) == 1:
        pts = pts.reshape(1, -1)
    for pt in pts:
        cv2.circle(img, (round(pt[0]), round(pt[1])), radius, color, -1)
    return img

def Draw3DPts(img, pts, ins_matrix, color=(0, 255, 0), radius=1):
    if len(pts.shape) == 1:
        pts = pts.reshape(1, -1)
    for pt in pts:
        pt_3d = np.dot(ins_matrix, pts.reshape(3, 1))[:3]
        pts_3d /= pts_3d[2]
        cv2.circle(img, (round(pt_3d[0]), round(pt_3d[1])), radius, color, -1)
    return img


class GazeGeneVisualizer_OriginalFaceCrops:
    def __init__(self, subject='subject1', camera='camera0'):
        self.subject = subject
        self.camera = camera
        self.base_dir = f'/home/byw/Dataset/GazeGene_FaceCrops/'
        self.complex_label_path = f'{self.base_dir}/{self.subject}/labels/complex_label_{self.camera}.pkl'
        self.gaze_label_path = f'{self.base_dir}/{self.subject}/labels/gaze_label_{self.camera}.pkl'
        with open(self.complex_label_path, 'rb') as f:
            self.complex_label = pickle.load(f)
        with open(self.gaze_label_path, 'rb') as f:
            self.gaze_label = pickle.load(f)
        self.current_idx = None
        self.current_img = None
        if not os.path.exists('./imgs'):
            os.makedirs('./imgs')

    def GetImg(self, idx):
        self.current_idx = idx
        img_path = f'{self.base_dir}/{self.subject}/{self.complex_label["img_path"][idx]}'
        self.current_img = cv2.imread(img_path)
        return self.current_img
    
    def DrawEyeballCenter2D(self, color=(0, 255, 0), length=4, radius=2):
        assert self.current_img is not None
        Draw_X(self.current_img, self.complex_label['eyeball_center_2D'][self.current_idx], length, color, radius)

    def DrawPupilCenter2D(self, color=(0, 255, 0), length=4, radius=2):
        assert self.current_img is not None
        Draw_X(self.current_img, self.complex_label['pupil_center_2D'][self.current_idx], length, color, radius)

    def DrawIrisMesh2D(self, color=(255, 255, 0), radius=2):
        assert self.current_img is not None
        iris_mesh_2D = self.complex_label['iris_mesh_2D'][self.current_idx]
        Draw2DPts(self.current_img, iris_mesh_2D[0], color=color, radius=radius)
        Draw2DPts(self.current_img, iris_mesh_2D[1], color=color, radius=radius)

    def DrawEyeballCenter3D(self, color=(0, 255, 0), length=4, radius=2):
        assert self.current_img is not None
        ins_matrix = self.complex_label['intrinsic_matrix_cropped'][self.current_idx]
        ins_matrix = np.tile(ins_matrix, (2, 1, 1))
        eyeball_center_2D = np.matmul(ins_matrix, self.complex_label['eyeball_center_3D'][self.current_idx].reshape(2, 3, 1)).reshape(2, 3)
        eyeball_center_2D /= eyeball_center_2D[:, 2].reshape(-1, 1)
        Draw_X(self.current_img, eyeball_center_2D[:, :2], length, color, radius)
    
    def DrawPupilCenter3D(self, color=(0, 255, 0), length=4, radius=2):
        assert self.current_img is not None
        ins_matrix = self.complex_label['intrinsic_matrix_cropped'][self.current_idx]
        ins_matrix = np.tile(ins_matrix, (2, 1, 1))
        pupil_center_2D = np.matmul(ins_matrix, self.complex_label['pupil_center_3D'][self.current_idx].reshape(2, 3, 1)).reshape(2, 3)
        pupil_center_2D /= pupil_center_2D[:, 2].reshape(-1, 1)
        Draw_X(self.current_img, pupil_center_2D[:, :2], length, color, radius)

    def DrawIrisMesh3D(self, color=(255, 255, 0), radius=2):
        ins_matrix = self.complex_label['intrinsic_matrix_cropped'][self.current_idx]
        ins_matrix = np.tile(ins_matrix, (100, 1, 1))
        iris_L_mesh_2D = np.matmul(ins_matrix, self.complex_label['iris_mesh_3D'][self.current_idx][0].reshape(100, 3, 1)).reshape(100, 3)
        iris_L_mesh_2D /= iris_L_mesh_2D[:, 2].reshape(-1, 1)
        Draw2DPts(self.current_img, iris_L_mesh_2D[:, :2], color=color, radius=radius)
        iris_R_mesh_2D = np.matmul(ins_matrix, self.complex_label['iris_mesh_3D'][self.current_idx][1].reshape(100, 3, 1)).reshape(100, 3)
        iris_R_mesh_2D /= iris_R_mesh_2D[:, 2].reshape(-1, 1)
        Draw2DPts(self.current_img, iris_R_mesh_2D[:, :2], color=color, radius=radius)

    def DrawHeadGaze(self, color=(0, 255, 0), length=5, thickness=2):
        assert self.current_img is not None
        ins_matrix = self.complex_label['intrinsic_matrix_cropped'][self.current_idx]
        head_gaze_3D = self.gaze_label['gaze_C'][self.current_idx]
        center = np.mean(self.complex_label['eyeball_center_3D'][self.current_idx], axis=0)
        DrawArrow(self.current_img, center, head_gaze_3D, length, ins_matrix, color, thickness)

    def DrawVisualAxis(self, color=(0, 0, 255), length=5, thickness=2):
        assert self.current_img is not None
        ins_matrix = self.complex_label['intrinsic_matrix_cropped'][self.current_idx]
        visual_axis_L_3D = self.gaze_label['visual_axis_L'][self.current_idx]
        center_L = self.complex_label['pupil_center_3D'][self.current_idx][0]
        DrawArrow(self.current_img, center_L, visual_axis_L_3D, length, ins_matrix, color, thickness)
        visual_axis_R_3D = self.gaze_label['visual_axis_R'][self.current_idx]
        center_R = self.complex_label['pupil_center_3D'][self.current_idx][1]
        DrawArrow(self.current_img, center_R, visual_axis_R_3D, length, ins_matrix, color, thickness)

    def DrawOpticAxis(self, color=(255, 0, 0), length=5, thickness=2):
        assert self.current_img is not None
        ins_matrix = self.complex_label['intrinsic_matrix_cropped'][self.current_idx]
        optic_axis_L_3D = self.gaze_label['optic_axis_L'][self.current_idx]
        center_L = self.complex_label['pupil_center_3D'][self.current_idx][0]
        DrawArrow(self.current_img, center_L, optic_axis_L_3D, length, ins_matrix, color, thickness)
        optic_axis_R_3D = self.gaze_label['optic_axis_R'][self.current_idx]
        center_R = self.complex_label['pupil_center_3D'][self.current_idx][1]
        DrawArrow(self.current_img, center_R, optic_axis_R_3D, length, ins_matrix, color, thickness)

    def DrawHeadPose(self, length=4, thickness=2):
        assert self.current_img is not None
        extend = 1
        ins_matrix = self.complex_label['intrinsic_matrix_cropped'][self.current_idx]
        center = self.gaze_label['head_T_vec'][self.current_idx]
        head_R_mat = self.gaze_label['head_R_mat'][self.current_idx]
        head_x = head_R_mat[:, 0]
        head_y = head_R_mat[:, 1]
        head_z = head_R_mat[:, 2]
        DrawArrow(self.current_img, center-head_x*extend, head_x, length, ins_matrix, (0, 0, 255), thickness)
        DrawArrow(self.current_img, center-head_y*extend, head_y, length, ins_matrix, (0, 255, 0), thickness)
        DrawArrow(self.current_img, center-head_z*extend, head_z, length, ins_matrix, (255, 0, 0), thickness)


    def run_example(self):
        count = 0
        for i in range(100, 1000, 100):
            self.GetImg(i)
            self.DrawEyeballCenter2D()
            self.DrawPupilCenter2D()
            self.DrawIrisMesh2D()
            cv2.imwrite(f'./imgs/{count}_2D.jpg', self.current_img)

            self.GetImg(i)
            self.DrawEyeballCenter3D()
            self.DrawPupilCenter3D()
            self.DrawIrisMesh3D()
            cv2.imwrite(f'./imgs/{count}_3D.jpg', self.current_img)

            self.GetImg(i)
            # self.DrawHeadGaze()
            # self.DrawVisualAxis()
            self.DrawOpticAxis()
            self.DrawHeadPose()
            cv2.imwrite(f'./imgs/{count}_vector.jpg', self.current_img)
            count += 1



if __name__ == '__main__':
    # Demonstration of how to use GazeGene annotations by visualize them on images.
    # Images will be saved in the folder './imgs/'
    vis = GazeGeneVisualizer_OriginalFaceCrops()
    vis.run_example()





