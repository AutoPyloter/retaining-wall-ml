import cv2
import glob
import os
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCREENSHOTS_DIR = "screenshots/annotated"
OUTPUT_VIDEO     = "timelapse.mp4"
FRAME_RATE       = 30  # frames per second


def _extract_scenario_index(filepath):
    """
    Extract the leading integer from a screenshot filename for sorting.

    Screenshot filenames follow the pattern:
        {scenario_index}_{tab}_{values}.png
    e.g. 42_stability_0.24_0.12_319.74.png → 42

    Returns float('inf') if no leading integer is found, placing
    unrecognised files at the end of the sorted list.
    """
    base_name  = os.path.basename(filepath)
    first_part = base_name.split('_')[0]
    try:
        return int(first_part)
    except ValueError:
        return float('inf')


def create_timelapse(
    screenshots_dir=SCREENSHOTS_DIR,
    output_video=OUTPUT_VIDEO,
    frame_rate=FRAME_RATE,
):
    """
    Compile all PNG screenshots in a directory into a timelapse MP4 video.

    Screenshots are sorted by their leading scenario index so that the video
    follows the same order as the GEO5 data generation run.

    Parameters
    ----------
    screenshots_dir : str   — Path to the folder containing PNG screenshots
    output_video    : str   — Output video filename (MP4)
    frame_rate      : int   — Frames per second for the output video

    Raises
    ------
    FileNotFoundError
        If no PNG files are found in screenshots_dir.
    """
    images = sorted(
        glob.glob(os.path.join(screenshots_dir, "*.png")),
        key=_extract_scenario_index,
    )

    if not images:
        raise FileNotFoundError(
            f"No PNG files found in '{screenshots_dir}'. "
            "Run the data generation pipeline first."
        )

    # Determine frame dimensions from the first image
    first_frame = cv2.imread(images[0])
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    for image_path in images:
        frame = cv2.imread(image_path)
        writer.write(frame)

    writer.release()
    cv2.destroyAllWindows()

    print(f"Timelapse saved: {output_video} ({len(images)} frames at {frame_rate} fps)")


if __name__ == "__main__":
    create_timelapse()
