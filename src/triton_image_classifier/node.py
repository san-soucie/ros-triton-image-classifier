import functools

import numpy as np
import rospy
import tritonclient.grpc as grpcclient

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from PIL import Image as PILIMage
import cv2

from .triton_api import (
    Model,
    ImageInput,
    ClassificationOutput,
    ScalingMode,
    initialize_model,
)
from triton_image_classifier.msg import Classification, ObjectHypothesisWithClassName


class ImageClassifier:
    """
    Wrapper around the Triton Inference Server client,
    """

    def __init__(self, model=None, url=None):
        self.client = grpcclient.InferenceServerClient(url=url)
        self.model = model

    def classify(self, image):
        pass


def on_image(model, class_pub, image_msg):
    # Use the cv_bridge to convert to an OpenCV image object
    rospy.logdebug("Processing image...")
    img = CvBridge().imgmsg_to_cv2(image_msg)
    color_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = PILIMage.fromarray(color_converted)
    # Ask the classifier to
    result = model.infer(pil_image).output
    if len(result) != 1 or len(result[0]) != 1:
        rospy.logerr("Unexpected result from classifier: %s", result)
        return
    header = Header()
    header.stamp = rospy.Time.now()
    results = ObjectHypothesisWithClassName(class_=result[0][0].class_name, score=result[0][0].score)
    msg = Classification(header=header, results=[results])
    class_pub.publish(msg)


def main():
    rospy.init_node("classifier", anonymous=True)
    rospy.loginfo("Started classifier node.")
    model = initialize_model(
        rospy.get_param("~triton_server_url"), rospy.get_param("~classifier_model")
    )
    rospy.loginfo("Initialized model")
    model.input = ImageInput(scaling=ScalingMode.INCEPTION)
    model.output = ClassificationOutput(classes=1)

    # Advertise that we will publish a "/class" subtopic of the image topic
    class_pub = rospy.Publisher(
        rospy.get_param("~topic") + "/class", Classification, queue_size=1
    )

    # Subscribe to the roi image data
    rospy.Subscriber(
        rospy.get_param("~topic") + "/roi/image",
        Image,
        functools.partial(on_image, model, class_pub),
    )

    rospy.spin()


if __name__ == "__main__":
    main()
