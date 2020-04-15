import face_recognition as fr
import numpy


class FaceFeatureRecognizer(object):
    def image_has_face(self, image: numpy.ndarray):
        return self.face_locations(image) != []

    def face_locations(self, image: numpy.ndarray):
        return fr.face_locations(image)
