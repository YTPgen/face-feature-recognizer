class Face(object):
    def __init__(self, landmarks: dict):
        self.chin = landmarks["chin"]
        self.left_eyebrow = landmarks["left_eyebrow"]
        self.right_eyebrow = landmarks["right_eyebrow"]
        self.nose_bridge = landmarks["nose_bridge"]
        self.nose_tip = landmarks["nose_tip"]
        self.left_eye = landmarks["left_eye"]
        self.right_eye = landmarks["right_eye"]
        self.top_lip = landmarks["top_lip"]
        self.bottom_lip = landmarks["bottom_lip"]

    def center_of(self, feature: list) -> tuple:
        """Returns the center of a facial feature.

        Args:
            feature (list): Outline of feature

        Returns:
            tuple: Center of feature (x,y)
        """
        average_x, average_y = 0, 0
        for e in feature:
            average_x += e[0] // len(feature)
            average_y += e[1] // len(feature)
        return (average_x, average_y)
