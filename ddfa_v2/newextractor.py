import os
import math
import gc
import tempfile
import pickle
from pathlib import Path
import uuid
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QSlider, QPushButton, 
                             QCheckBox, QFileDialog, QTabWidget, QLineEdit, QHBoxLayout, QInputDialog, 
                             QMessageBox, QSizePolicy, QComboBox, QSpinBox, QGroupBox, QFormLayout)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt
from qt_material import apply_stylesheet
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.metrics import pairwise_distances, silhouette_score
import torch
from TDDFA import TDDFA
from FaceBoxes import FaceBoxes
from dface import FaceNet
from tqdm import tqdm
import yaml
from kneed import KneeLocator
from DFLIMG import DFLJPG
from mainscripts.Extractor import ExtractSubprocessor, FaceType, LandmarksProcessor
from mainscripts import XSegUtil
from core.leras import nn


device = 'cuda' if torch.cuda.is_available() else 'cpu'
config_path = os.path.join(os.path.dirname(__file__), 'configs', 'mb1_120x120.yml')
cfg = yaml.load(open(config_path), Loader=yaml.SafeLoader)
tddfa = TDDFA(gpu_mode=device, **cfg)
face_boxes = FaceBoxes()
facenet = FaceNet(device=device)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Clustering GUI")
        self.setGeometry(100, 100, 800, 600)
        self.selected_cluster = -1
        self.clustering_method = "DBSCAN"
        self.image_size = 512 
        self.crop_type = "Whole Face"
        self.png_output = "False"
        self.frames = 1
        self.currentPage = 0
        self.totalPages = 0
        self.image_size = 512
        self.fps = 0
        self.setupUI()
        self.connectActions()

    def setupUI(self):
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.layout = QVBoxLayout()
        self.centralWidget.setLayout(self.layout)

        self.tabWidget = QTabWidget()
        self.layout.addWidget(self.tabWidget)
        self.tabWidget.currentChanged.connect(self.onTabChanged)

        self.mainTab = QWidget()
        self.settingsTab = QWidget()
        self.epsGraphTab = QWidget()

        self.tabWidget.addTab(self.mainTab, "Main")
        self.tabWidget.addTab(self.settingsTab, "Settings")
        self.tabWidget.addTab(self.epsGraphTab, "eps Graph")

        self.mainLayout = QVBoxLayout()
        self.mainTab.setLayout(self.mainLayout)

        self.imageLabel = QLabel()
        self.mainLayout.addWidget(self.imageLabel)
        self.centralWidget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.selectVideoButton = QPushButton("Select Video")
        self.mainLayout.addWidget(self.selectVideoButton)

        self.extractButton = QPushButton("Extract")
        self.mainLayout.addWidget(self.extractButton)
        
        self.exportButton = QPushButton("Export Selected Cluster")
        self.mainLayout.addWidget(self.exportButton)

        self.epsInput = QLineEdit()
        self.epsInput.setPlaceholderText("Enter eps value")
        self.epsInput.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

        self.refreshClustersButton = QPushButton("Refresh Clusters")

        epsLayout = QHBoxLayout()
        epsLayout.addWidget(self.epsInput)
        epsLayout.addWidget(self.refreshClustersButton)
        self.mainLayout.addLayout(epsLayout)


        self.prevPageButton = QPushButton("Previous Page")
        self.nextPageButton = QPushButton("Next Page")

        navigationLayout = QHBoxLayout()
        navigationLayout.addWidget(self.prevPageButton)
        navigationLayout.addWidget(self.nextPageButton)
        self.mainLayout.addLayout(navigationLayout)

        self.createSettingsTab()
        self.createEpsGraphTab()

    def createSettingsTab(self):
        self.settingsLayout = QVBoxLayout()
        self.settingsTab.setLayout(self.settingsLayout)

        imageSizeGroupBox = QGroupBox("Image Settings")
        imageSizeLayout = QFormLayout()

        self.imageSizeSpinBox = QSpinBox()
        self.imageSizeSpinBox.setMinimum(32)
        self.imageSizeSpinBox.setMaximum(2048)
        self.imageSizeSpinBox.setValue(self.image_size)
        self.imageSizeSpinBox.valueChanged.connect(self.updateImageSize)
        imageSizeLayout.addRow(QLabel("Image Size:"), self.imageSizeSpinBox)
        
        self.cropTypeComboBox = QComboBox()
        self.cropTypeComboBox.addItems(["Whole Face", "Mid Face", "Full Face", "Head"])
        imageSizeLayout.addRow(QLabel("Crop Type:"), self.cropTypeComboBox)

        imageSizeGroupBox.setLayout(imageSizeLayout)
        self.settingsLayout.addWidget(imageSizeGroupBox)

        exportGroupBox = QGroupBox("Export Settings")
        exportLayout = QFormLayout()
        
        self.pngOutputCheckBox = QCheckBox("PNG Output")
        exportLayout.addRow(self.pngOutputCheckBox)

        self.framesSpinBox = QSpinBox()
        self.framesSpinBox.setMinimum(1)
        self.framesSpinBox.setValue(1)
        exportLayout.addRow(QLabel("FPS (How many frames per second will be extracted):"), self.framesSpinBox)

        exportGroupBox.setLayout(exportLayout)
        self.settingsLayout.addWidget(exportGroupBox)

        clusteringGroupBox = QGroupBox("Clustering Settings")
        clusteringLayout = QFormLayout()

        self.clusteringComboBox = QComboBox()
        self.clusteringComboBox.addItems(["DBSCAN", "OPTICS"])
        clusteringLayout.addRow(QLabel("Clustering Method:"), self.clusteringComboBox)

        clusteringGroupBox.setLayout(clusteringLayout)
        self.settingsLayout.addWidget(clusteringGroupBox)

        self.updateSettingsButton = QPushButton("Update Settings", self)
        self.settingsLayout.addWidget(self.updateSettingsButton)


    def createEpsGraphTab(self):
        self.epsGraphLayout = QVBoxLayout()
        self.epsGraphTab.setLayout(self.epsGraphLayout)

        self.epsGraphLabel = QLabel()
        self.epsGraphLabel.setPixmap(QPixmap('eps_analysis.png'))
        self.epsGraphLayout.addWidget(self.epsGraphLabel)
        self.epsGraphLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def updateImageSize(self, value):
        self.image_size = value

    def updateSettings(self):
        self.image_size = self.imageSizeSpinBox.value()
        self.crop_type = self.cropTypeComboBox.currentText()
        self.png_output = self.pngOutputCheckBox.isChecked()
        self.fps = self.framesSpinBox.value()
        self.clustering_method = self.clusteringComboBox.currentText()


    def connectActions(self):
        self.extractButton.clicked.connect(self.extract)
        self.selectVideoButton.clicked.connect(self.selectVideo)
        self.exportButton.clicked.connect(self.export_cluster)
        self.updateSettingsButton.clicked.connect(self.updateSettings)
        self.prevPageButton.clicked.connect(self.prevPage)
        self.nextPageButton.clicked.connect(self.nextPage)
        self.refreshClustersButton.clicked.connect(self.refreshClusters)

    def closeEvent(self, event):
        if hasattr(self, 'temp_file'):
            os.remove(self.temp_file.name)
        event.accept()

    def onTabChanged(self, index):
        if index == 0:  # Main tab
            if hasattr(self, 'imageLabel'):
                self.imageLabel.setVisible(True)
            self.centralWidget.setMaximumSize(842, 1092)
            self.resize(842, 1092)
        elif index == 1:  # Settings tab
            if hasattr(self, 'imageLabel'):
                self.imageLabel.setVisible(False)
            self.centralWidget.setMaximumSize(682, 552)
            self.resize(682, 552)
        elif index == 2:  # eps Graph tab
            if hasattr(self, 'imageLabel'):
                self.imageLabel.setVisible(False)
            self.centralWidget.setMaximumSize(682, 552)
            self.resize(682, 552)
            self.epsGraphLabel.setPixmap(QPixmap('eps_analysis.png'))

        self.adjustSize()
        self.update()
        self.layout.update()
        self.centralWidget.update()
        

    def selectVideo(self):
        video_path = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi);;All Files (*)")[0]
        if not video_path:
            print("No video selected.")
            return
        else:
            self.video_path = video_path
            print(f"Video path: {self.video_path}")

    def selectFolder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", "")
        if not folder_path:
            print("No folder selected.")
            return
        else:
            self.folder_path = folder_path

    def mousePressEvent(self, event):
        if self.tabWidget.currentIndex() == 0 and hasattr(self, 'clusters'):
            if event.button() == Qt.LeftButton:
                self.selectCluster(event.x(), event.y())

    def selectCluster(self, x, y):
        cluster_width, cluster_height = 160, 160
        y -= 50
        y = max(y, 0)
        row = y // cluster_height
        clusters_per_page = 5
        row = min(row, clusters_per_page - 1)
        self.selected_cluster = int(row + (self.currentPage * clusters_per_page))
        self.highlightSelectedCluster()


    def highlightSelectedCluster(self):
        pixmap = self.basePixmap.copy()
        painter = QPainter(pixmap)
        painter.setPen(QPen(Qt.red, 5))
        clusters_per_page = 5
        row = self.selected_cluster - (self.currentPage * clusters_per_page)
        painter.drawRect(0, row * 160, 5 * 160, 160)
        painter.end()
        self.imageLabel.setPixmap(pixmap)
            
    def extract(self):
        temp_face_data_file = tempfile.NamedTemporaryFile(delete=False)
        temp_embedding_file = tempfile.NamedTemporaryFile(delete=False)

        if not hasattr(self, "video_path"):
            QMessageBox.warning(self, "No Video", "No video selected to extract from.")
            return

        for actual_frame_number, frame in get_frames(self.video_path, self.fps):
            faces, params, roi_box_lst = detect_faces([frame], tddfa, face_boxes)
            landmarks = get_landmarks(params, roi_box_lst)

            for face_index, (face, landmark) in enumerate(zip(faces, landmarks)):
                face_data_entry = {
                    'frame_index': actual_frame_number,
                    'face_index': face_index,
                    'face': face,
                    'landmark': landmark
                }
                pickle.dump(face_data_entry, temp_face_data_file)

        with open(temp_face_data_file.name, 'rb') as file:
            self.face_data = []
            while True:
                try:
                    self.face_data.append(pickle.load(file))
                except EOFError:  # No more data to read
                    break

        self.all_faces = [data['face'] for data in self.face_data]
        self.embeds = generate_embedding(self.all_faces, temp_embedding_file.name)
        
        with open(temp_embedding_file.name, 'rb') as file:
            self.embeds = []
            while True:
                try:
                    self.embeds.append(pickle.load(file))
                except EOFError:
                    break

        eps = get_best_eps(self.embeds)
        if eps == None:
            eps = 0.35
            print(f"No best eps found, using default value of {eps}.")
        self.labels = cluster_faces(self.embeds, self.clustering_method, eps)
        for i, data in enumerate(self.face_data):
            data['cluster'] = self.labels[i]
        self.adjustSize()
        self.display_clusters(self.all_faces, self.labels)
        return self.labels, self.embeds
    
    def refreshClusters(self):
        self.epsInput.text()
        self.labels = cluster_faces(self.embeds, self.clustering_method, float(self.epsInput.text()))
        for i, data in enumerate(self.face_data):
            data['cluster'] = self.labels[i]
        self.display_clusters(self.all_faces, self.labels)
        
    
    def display_clusters(self, faces, labels):
        if hasattr(self, 'basePixmap'):
            del self.basePixmap
            gc.collect()

        if len(faces) != len(labels):
            print(f"Number of faces: {len(faces)}")
            print(f"Number of labels: {len(labels)}")
            raise ValueError("The number of faces does not match the number of labels.")

        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:  # Noise label in DBSCAN
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(faces[i])

        self.clusters = clusters


        page_rows = 5
        canvas_width = 5

        canvas = np.zeros((page_rows * 160, canvas_width * 160, 3), dtype=np.uint8)

        start_idx = self.currentPage * page_rows
        end_idx = min(start_idx + page_rows, len(clusters))


        for idx, (label, cluster_faces) in enumerate(list(clusters.items())[start_idx:end_idx]):
            cluster_indices = np.linspace(0, len(cluster_faces) - 1, 5, dtype=int)
            for j, index in enumerate(cluster_indices):
                col = j
                row = idx % page_rows
                if index < len(cluster_faces):
                    resized_face = cv2.resize(cluster_faces[index], (160, 160))
                    canvas[row * 160:(row + 1) * 160, col * 160:(col + 1) * 160] = resized_face

        height, width, channel = canvas.shape
        bytes_per_line = 3 * width
        q_image = QImage(canvas.data, width, height, bytes_per_line, QImage.Format_RGB888)

        self.selected_cluster = -1
        self.imageLabel.setPixmap(QPixmap.fromImage(q_image))
        pixmap = QPixmap.fromImage(q_image)

        self.selected_cluster = -1
        self.imageLabel.setPixmap(pixmap)
    
        self.basePixmap = QPixmap.fromImage(q_image)

    def prevPage(self):
        if not hasattr(self, "clusters"):
            return
        self.totalPages = math.ceil(len(self.clusters) / 5)
        print(f"Current page: {self.currentPage}")
        if self.currentPage > 0:
            self.currentPage -= 1
            self.display_clusters(self.all_faces, self.labels)

    def nextPage(self):
        if not hasattr(self, "clusters"):
            return
        self.totalPages = math.ceil(len(self.clusters) / 5)
        print(f"Current page: {self.currentPage}")
        if self.currentPage < self.totalPages - 1:
            self.currentPage += 1
            self.display_clusters(self.all_faces, self.labels)




    def export_cluster(self):
        if not self.face_data:
            QMessageBox.warning(self, "No Data", "Face and landmark data is not available.")
            return
        if self.selected_cluster == -1:
            QMessageBox.warning(self, "No Cluster Selected", "Please select a cluster first.")
            return

        if not hasattr(self, 'clusters'):
            QMessageBox.warning(self, "No Clusters", "No clusters available for export.")
            return

        cluster_faces = self.clusters.get(self.selected_cluster, [])
        if not cluster_faces:
            QMessageBox.warning(self, "Empty Cluster", "Selected cluster is empty.")
            return

        if self.crop_type == "Whole Face":
            self.crop_type = FaceType.WHOLE_FACE
        elif self.crop_type == "Mid Face":
            self.crop_type = FaceType.MID_FULL
        elif self.crop_type == "Full Face":
            self.crop_type = FaceType.FULL
        elif self.crop_type == "Head":
            self.crop_type = FaceType.HEAD
        else:
            self.crop_type = FaceType.WHOLE_FACE

        os.makedirs("faces", exist_ok=True)

        for data in self.face_data:
            frame_index = data['frame_index']
            face_index = data['face_index']
            face = data['face']
            landmark = data.get('landmark', [])
            cluster = data.get('cluster', -1)

            if cluster != self.selected_cluster or face_index != 0:
                continue

            relative_path = os.path.join("../../..", "workspace")
            filename = os.path.join(relative_path, "faces", f"cluster_{self.selected_cluster}/{frame_index}_{uuid.uuid4()}_{face_index}.jpg")
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            if self.fps == 0:
                cap = cv2.VideoCapture(self.video_path)
                actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
                cap.release()
            else:
                actual_fps = self.fps

            time_in_seconds = frame_index / actual_fps
            frame = get_frame(self.video_path, time_in_seconds, self.fps)   
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image_to_face_mat = LandmarksProcessor.get_transform_mat(landmark, self.image_size, self.crop_type)
            face_image = cv2.warpAffine(frame, image_to_face_mat, (self.image_size, self.image_size), cv2.INTER_LANCZOS4)
            face_image = Image.fromarray(face_image)
            face_image_landmarks = LandmarksProcessor.transform_points(landmark, image_to_face_mat)

            face_image.save(filename)

            dflimg = DFLJPG.load(filename)
            dflimg.set_landmarks(face_image_landmarks.tolist())
            rect = LandmarksProcessor.get_rect_from_landmarks(landmark)
            dflimg.set_face_type(FaceType.toString(self.crop_type))

            dflimg.set_source_filename(f"{str(frame_index).zfill(5)}.jpg")
            dflimg.set_source_rect(rect)
            dflimg.set_source_landmarks(landmark)
            dflimg.set_image_to_face_mat(image_to_face_mat)
            dflimg.save()

        QMessageBox.information(self, "Exported", "Faces exported to workspace/faces folder.")


def get_frames(video_path, fps=0):
    cap = cv2.VideoCapture(video_path)
    
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    if fps == 0:
        frame_interval = 1
    else:
        frame_interval = original_fps // fps
    
    frame_count = 0
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // frame_interval, desc="Extracting...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame_count % frame_interval == 0:
            actual_frame_number = frame_count
            yield (actual_frame_number, frame)
            pbar.update(1)
        frame_count += 1
    
    cap.release()

def get_frame(video_path, time_index, fps=0):
    cap = cv2.VideoCapture(video_path)
    
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    if fps == 0:
        fps = original_fps

    frame_index_to_extract = int(time_index * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index_to_extract)
    
    # Read the frame at that position
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return frame
    else:
        return None

def detect_faces(frames, tddfa, face_boxes):
    faces = []
    params = []

    for frame in frames:
        h, w, _ = frame.shape
        boxes = face_boxes(frame)
        if boxes is not None:
            param_lst, roi_box_lst = tddfa(frame, boxes)
            for box, param in zip(boxes, param_lst):
                probability = box[-1]
                if probability >= 0.95:
                    x1, y1, x2, y2 = get_boundingbox(box, w, h)
                    face = frame[y1:y2, x1:x2]
                    faces.append(face)
                    params.append(param)

    return faces, params, roi_box_lst

def get_landmarks(param_lst, roi_box_lst):
    """
    Extract landmarks from a face.
    :param face: A single face image.
    :return: Landmarks for the face.
    """
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
    all_landmarks = []
    for vers in ver_lst:
        landmarks = [(ver[0], ver[1]) for ver in ver_lst[0].T]
        all_landmarks.append(landmarks)

    return all_landmarks

def get_best_eps(embeds, eps_range=(0.1, 0.9), eps_step=0.05, dbscan_params=None):
    """
    Determine the best 'eps' value for DBSCAN clustering based on the number of noise points.
    
    Parameters:
    - embeds: array-like, shape = [n_samples, n_features]
      Data embeddings.
    - eps_range: tuple (start, end), default = (0.1, 0.9)
      Range of 'eps' values to test.
    - eps_step: float, default = 0.05
      Increment step of 'eps' values.
    - dbscan_params: dict, optional
      Additional parameters for DBSCAN (e.g., min_samples, metric).

    Returns:
    - Best 'eps' value.
    """
    if dbscan_params is None:
        dbscan_params = {'metric': 'cosine', 'min_samples': 5}

    eps_values = np.arange(eps_range[0], eps_range[1], eps_step)
    noise_points = []
    silhouette_scores = []

    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, **dbscan_params)
        labels = dbscan.fit_predict(embeds)
        
        noise = np.sum(labels == -1)
        noise_points.append(noise)
        
        if len(set(labels)) > 1:
            silhouette_scores.append(silhouette_score(embeds, labels, metric=dbscan_params['metric']))
        else:
            silhouette_scores.append(-1)

    plt.figure(figsize=(6, 5))
    plt.plot(eps_values, noise_points, marker='o')
    plt.title('Noise points vs. eps value')
    plt.xlabel('eps')
    plt.ylabel('Noise points')
    kneedle_noise = KneeLocator(eps_values, noise_points, curve='convex', direction='decreasing')
    plt.vlines(kneedle_noise.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')

    plt.tight_layout()
    plt.savefig('eps_analysis.png')

    print(f"Best eps based on noise: {kneedle_noise.knee}")
    return kneedle_noise.knee


def cluster_faces(embeds, clustering_method, eps):
    print("Clustering faces.")
    if clustering_method == "DBSCAN":
        dbscan = DBSCAN(eps=eps, metric='cosine', min_samples=5)
        labels = dbscan.fit_predict(embeds)
    elif clustering_method == "OPTICS":
        optics = OPTICS(min_samples=5, metric='cosine')
        labels = optics.fit_predict(embeds)
    print(f"Labels: {labels}")

    print(f"Unique labels (clusters): {np.unique(labels)}")
    return labels

def get_boundingbox(box, w, h, scale=1.3):
    x1, y1, x2, y2, _ = box
    size = int(max(x2-x1, y2-y1) * scale)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    if size > w or size > h:
        size = int(max(x2-x1, y2-y1))
    x1 = max(int(center_x - size // 2), 0)
    y1 = max(int(center_y - size // 2), 0)
    size = min(w - x1, size)
    size = min(h - y1, size)
    return x1, y1, x1 + size, y1 + size


def generate_embedding(all_faces, temp_embedding_filepath):
    embeddings = facenet.embedding(all_faces)
    
    with open(temp_embedding_filepath, 'wb') as file:
        for embedding in embeddings:
            pickle.dump(embedding, file)
    
    return temp_embedding_filepath


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    apply_stylesheet(app, theme='dark_teal.xml')
    window.show()
    app.exec_()