import face_recognition as fr
import numpy
from face_feature_recognizer.face import Face


class FaceFeatureRecognizer(object):
    # TODO: Use batch processing for face locations

    @staticmethod
    def image_has_face(image: numpy.ndarray):
        return FaceFeatureRecognizer.face_locations(image) != []

    @staticmethod
    def face_locations(image: numpy.ndarray):
        """Finds all face locations in an image

        Returns:
            tuple: Face box (top, right, bottom, left)
        """

        return fr.face_locations(image)

    @staticmethod
    def load_image(path: str) -> numpy.ndarray:
        return fr.load_image_file(path)

    @staticmethod
    def face_centers(image: numpy.ndarray):
        faces = FaceFeatureRecognizer.face_locations(image)
        return [FaceFeatureRecognizer.face_to_center(face) for face in faces]

    @staticmethod
    def face_to_center(face: tuple) -> tuple:
        """Converts a face box tuple into a center tuple

        Args:
            face (tuple): Face box (top, right, bottom, left)

        Returns:
            tuple: Face center (x, y)
        """

        return ((face[1] + face[3]) / 2, (face[0] + face[2]) / 2)

    @staticmethod
    def find_faces(image: numpy.ndarray) -> list:
        """Finds information about all faces in an image.

        Args:
            image (numpy.ndarray): Image

        Returns:
            list: Face objects with facial feature information
        """
        landmarks = fr.face_landmarks(image)
        return [Face(l) for l in landmarks]
