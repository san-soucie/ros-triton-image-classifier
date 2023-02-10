import functools

import numpy as np
import rospy
import tritonclient.grpc as grpcclient

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from .msg import Classification, ObjectHypothesisWithClassName


class ImageClassifier:
    '''
    Wrapper around the Triton Inference Server client, 
    '''
    def __init__(self, model=None, url=None):
        self.client = grpcclient.InferenceServerClient(url=url)
        self.model = model
    
    def classify(self, image):
        pass


def on_image(classifier, class_pub, image_msg):
    # Use the cv_bridge to convert to an OpenCV image object
    img = CvBridge().imgmsg_to_cv2(image_msg)
    
    # Ask the classifier to 
    


def main():
    rospy.init_node('classifier', anonymous=True)

    # Connect to the Triton Inference Server
    classifier = ImageClassifier(
        url=rospy.get_param('~server'),
        model=rospy.get_param('~model'),
    )

    # Advertise that we will publish a "/class" subtopic of the image topic
    class_pub = rospy.Publisher(
        rospy.get_param('~topic') + '/class',
        Classification,
        queue_size=1
    )

    # Subscribe to the raw image data
    rospy.Subscriber(
        rospy.get_param('~topic') + '/raw',
        Image,
        functools.partial(on_image, classifier, class_pub)
    )

    rospy.spin()


if __name__ == '__main__':
    main()
