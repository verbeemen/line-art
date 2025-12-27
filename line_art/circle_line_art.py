import random
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm

from line_art import config

IMAGE_NAME = "cat.png"
DATETIME_NOW = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
D = None  # width or the height of the image
N = 200
SKIP_RANGE = 10  # points, 5 degrees above or below from the current point will be skipped
INVERT_COLOUR = False
INTENSITY = 7


def read_png_image(image_path: str, channel: int = 0) -> np.ndarray:
    """
    Read a image, take a specific channal for gray scale conversion. and return as numpy array.
    Args:
        image_path (str): path to image file.
        channel (int): channel index to use for gray scale conversion. Default is 0 (Red channel).
    """
    return np.array(Image.open(image_path), dtype=np.int16)[:, :, channel]


def calculate_angle_between_two_points(
    center: tuple[int | float, int | float], p1: tuple[int, int], p2: tuple[int, int]
) -> float:
    """
    Computes the angle in degrees between two points on a circle relative to a given center.

    The function treats the points as vectors originating from the center and uses the
    dot-product formula to determine the angle between those vectors:
        v1 · v2 = |v1| * |v2| * cos(theta)

    After computing the cosine value, the angle is obtained using the arccosine function.
    Due to the mathematical definition of arccos, the resulting angle is always within the
    range 0° to 180°, representing the smallest possible angle between the two vectors.
    This is a property of the arccosine function, not a limitation of the circle.

    Numerical clipping is applied to the cosine value to avoid floating-point rounding
    issues that could produce invalid values (e.g., slightly above 1.0).

    The order of p1 and p2 does not affect the result.
    """

    # Vector from center to p1
    v1_x = p1[0] - center[0]
    v1_y = p1[1] - center[1]

    # Vector from center to p2
    v2_x = p2[0] - center[0]
    v2_y = p2[1] - center[1]

    # Dot product
    dot_product = v1_x * v2_x + v1_y * v2_y

    # Magnitudes (radius)
    norm_v1 = (v1_x**2 + v1_y**2) ** 0.5
    norm_v2 = (v2_x**2 + v2_y**2) ** 0.5

    # Calculate angle
    # Ensure denominator is not zero
    if norm_v1 * norm_v2 == 0:
        return 0

    # Calculate cosine of the angle using dot product
    # Formula: v1 . v2 = |v1| * |v2| * cos(theta)
    cosine_angle = dot_product / (norm_v1 * norm_v2)

    # Clip value to [-1, 1] to correct potential floating point errors
    # (e.g., 1.0000000002 becoming NaN in arccos)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # Calculate angle in radians and convert to degrees
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)

    return round(float(angle_deg), 7)


def calculate_line_coordinates(
    point_a: tuple[int, int], point_b: tuple[int, int], intensity: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bresenham's line algorithm geoptimaliseerd voor Numpy indices (y, x)."""
    # get the start and end points
    x0, y0 = point_a
    x1, y1 = point_b

    # Bresenham's line algorithm
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    xs = []
    ys = []
    values = []
    while True:
        # blend the pixel value on the canvas with the image pixel value
        xs.append(x0)
        ys.append(y0)
        values.append(intensity)

        if x0 == x1 and y0 == y1:
            break
        err2 = 2 * err
        if err2 > -dy:
            err -= dy
            x0 += sx
        if err2 < dx:
            err += dx
            y0 += sy
    return np.array(xs), np.array(ys), np.array(values, dtype=np.int16)


def draw_lines(
    np_img: np.ndarray,
    line_coordinates: dict[tuple[int, int], dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]],
    invert_colour: bool = False,
) -> np.ndarray:
    """Create line art by drawing lines between given points on the image."""
    # create an empty canvas
    canvas = np.zeros_like(np_img, dtype=np.int16)
    sign = 1
    if invert_colour:
        canvas = 255 - canvas
        sign = -1

    # select a random point to start with
    current_point = random.choice(list(line_coordinates))
    previous_point = (-1, -1)
    stop = 40
    print(f"Startpunt: {current_point}")

    for e in tqdm(range(100_000)):
        # draw lines from the current point to all other points and select the one with the highest score
        xs_best = None
        ys_best = None
        values_best = 0
        best_score = 0
        next_point = None
        for tmp_point, (xs, ys, values) in line_coordinates[current_point].items():
            if tmp_point == previous_point:
                continue

            # calculate the current pixel distance
            current_pixel_distance = np.sum(np.abs(np_img[xs, ys] - canvas[xs, ys]))

            # calculate the new pixel values after drawing the line
            new_pixels = np.clip(canvas[xs, ys] + sign * values, 0, 255)

            # calculate the new pixel distance
            new_pixel_distance = np.sum(np.abs(np_img[xs, ys] - new_pixels))

            # calculate the score as the improvement in pixel distance
            if (e % 2) == 0:
                score = current_pixel_distance / len(xs) - new_pixel_distance / len(xs)
            else:
                score = current_pixel_distance - new_pixel_distance

            if score > best_score:
                best_score = score
                xs_best = xs
                ys_best = ys
                values_best = new_pixels
                next_point = tmp_point

        # If we didn't find any line that improves the image, break or skip
        if next_point is None:
            print("No improvement found, skipping to a new random point.")
            current_point = random.choice(list(line_coordinates))
            previous_point = (-1, -1)
            stop -= 1
            continue

        # print(f"Drew line from {current_point} to {next_point} with score {best_score:.2f}")
        canvas[xs_best, ys_best] = values_best
        previous_point = current_point
        current_point = next_point

        # save the image
        if e % 2500 == 0:
            output_filename = f"{IMAGE_NAME.split('.')[0]}_line_art_{DATETIME_NOW}_{e}.png"
            Image.fromarray(canvas.astype(np.uint8)).save(config.OUTPUT_DIR / output_filename, format="PNG")

    return canvas


def main() -> None:
    # read the image
    np_img = read_png_image(config.INPUT_DIR / IMAGE_NAME, channel=0)

    # get the image dimensions and calculate the circle parameters
    height, width = np_img.shape
    r = min(width, height) // 2 - 1 if D is None else D
    mid_x = width / 2
    mid_y = height / 2

    # calulate the circle points
    points = [
        (
            int(mid_x + r * np.cos(2 * np.pi * i / N)),
            int(mid_y + r * np.sin(2 * np.pi * i / N)),
        )
        for i in range(N)
    ]

    # calculate the line coordinates between all points
    line_coordinates = {}
    for point_x in points:
        for point_y in points:
            # skip points that are too close to each other
            if (calculate_angle_between_two_points((mid_x, mid_y), point_x, point_y) < (SKIP_RANGE / 2)) and (
                SKIP_RANGE <= 0
            ):
                continue
            line_coordinates.setdefault(point_x, {})[point_y] = calculate_line_coordinates(point_x, point_y, INTENSITY)

    # draw the lines
    output = draw_lines(np_img, line_coordinates, invert_colour=INVERT_COLOUR)

    # write the line as a new image
    output_filename = f"{IMAGE_NAME.split('.')[0]}_line_art_{DATETIME_NOW}.png"
    Image.fromarray(output.astype(np.uint8)).save(config.OUTPUT_DIR / output_filename, format="PNG")


if __name__ == "__main__":
    main()
