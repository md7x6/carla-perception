import cv2


def draw_bbox(image, bbox, label=None):
    """
    Draw a 2D bounding box on an image.

    Args:
        image: OpenCV image (BGR).
        bbox: Tuple of (x_min, y_min, x_max, y_max).
        label: Optional text to draw above the box.

    Returns:
        Image with the bounding box drawn.
    """

    x_min, y_min, x_max, y_max = bbox

    cv2.rectangle(
        image,
        (x_min, y_min),
        (x_max, y_max),
        (0, 255, 0),
        2,
    )

    if label:
        cv2.putText(
            image,
            label,
            (x_min, max(20, y_min - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return image