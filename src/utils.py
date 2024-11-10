def get_color(index: int):
    """
    Get correct color according to facial feature type
    Args:
        index: landmark index

    Returns:
        the corresponding BGR color (opencv standard)
    """
    if index < 33:
        return [255, 0, 0]
    elif 33 <= index < 51:
        return [0, 255, 0]
    elif 51 <= index < 60:
        return [255, 255, 0]
    elif 60 <= index < 76:
        return [255, 0, 255]
    elif 76 <= index < 88:
        return [0, 255, 255]
    elif 88 <= index < 96:
        return [0, 0, 255]
    else:
        return [0, 255, 255]
