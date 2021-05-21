from mtcnn import MTCNN
import numpy as np

face_cords_hist = {}

detector = MTCNN()


def select_face(frame, face):
    # increase face window
    x = face[0] - 1 * face[0] // 5
    y = face[1] - 1 * face[1] // 5
    w = face[2] + 2 * face[2] // 5
    h = face[3] + 2 * face[3] // 5

    return frame[y:y + h, x:x + w]


def process_face_on_frame(example: str, frame: np.ndarray, max_face_h, max_face_w):
    faces = detector.detect_faces(frame)

    if len(faces) == 1:
        face = faces[0]['box']
        frame = select_face(frame, face)
        face_cords_hist[example] = [face[0], face[1], face[2], face[3]]
        return frame
    elif len(faces) > 1:
        face_size = 0
        current_face = faces[0]['box']
        nearest_face = None
        for face in faces:
            current_face_size = current_face[2] * current_face[3]
            if current_face_size > face_size:
                face_size = current_face_size
                nearest_face = face['box']
        return select_face(frame, nearest_face)
    elif face_cords_hist.get(example) and len(face_cords_hist.get(example)) != 0:
        return select_face(frame, face_cords_hist.get(example))
    else:
        return np.zeros((max_face_h, max_face_w, 3))
