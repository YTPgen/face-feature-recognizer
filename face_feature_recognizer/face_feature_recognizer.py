import face_recognition as fr
import numpy


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
