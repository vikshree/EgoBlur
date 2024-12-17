import argparse
import os
from functools import lru_cache
from typing import List

import cv2
import numpy as np
import torch
import torchvision
from moviepy.editor import ImageSequenceClip
from moviepy.video.io.VideoFileClip import VideoFileClip
import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--face_model_path",
        required=False,
        type=str,
        default=None,
        help="Absolute EgoBlur face model file path",
    )

    parser.add_argument(
        "--face_model_score_threshold",
        required=False,
        type=float,
        default=0.9,
        help="Face model score threshold to filter out low confidence detections",
    )

    parser.add_argument(
        "--lp_model_path",
        required=False,
        type=str,
        default=None,
        help="Absolute EgoBlur license plate model file path",
    )

    parser.add_argument(
        "--lp_model_score_threshold",
        required=False,
        type=float,
        default=0.9,
        help="License plate model score threshold to filter out low confidence detections",
    )

    parser.add_argument(
        "--nms_iou_threshold",
        required=False,
        type=float,
        default=0.3,
        help="NMS iou threshold to filter out low confidence overlapping boxes",
    )

    parser.add_argument(
        "--scale_factor_detections",
        required=False,
        type=float,
        default=1,
        help="Scale detections by the given factor to allow blurring more area, 1.15 would mean 15% scaling",
    )

    parser.add_argument(
        "--hammerhead_folder",
        required=False,
        type=str,
        default=None,
        help="Absolute path of hammeread run results",
    )

    # parser.add_argument(
    #     "--input_image_folder",
    #     required=False,
    #     type=str,
    #     default=None,
    #     help="Absolute path for the folder where images are located on which we want to run detector",
    # )

    # parser.add_argument(
    #     "--output_image_folder",
    #     required=False,
    #     type=str,
    #     default=None,
    #     help="Absolute path where we want to store the visualized images",
    # )

    parser.add_argument(
        "--input_video_path",
        required=False,
        type=str,
        default=None,
        help="Absolute path for the given video on which we want to make detections",
    )

    parser.add_argument(
        "--output_video_path",
        required=False,
        type=str,
        default=None,
        help="Absolute path where we want to store the visualized video",
    )

    parser.add_argument(
        "--output_video_fps",
        required=False,
        type=int,
        default=30,
        help="FPS for the output video",
    )

    return parser.parse_args()


def create_output_directory(file_path: str) -> None:
    """
    parameter file_path: absolute path to the directory where we want to create the output files
    Simple logic to create output directories if they don't exist.
    """
    print(
        f"Directory {os.path.dirname(file_path)} does not exist. Attempting to create it..."
    )
    os.makedirs(os.path.dirname(file_path))
    if not os.path.exists(os.path.dirname(file_path)):
        raise ValueError(
            f"Directory {os.path.dirname(file_path)} didn't exist. Attempt to create also failed. Please provide another path."
        )


def validate_inputs(args: argparse.Namespace) -> argparse.Namespace:
    """
    parameter args: parsed arguments
    Run some basic checks on the input arguments
    """
    # input args value checks
    if not 0.0 <= args.face_model_score_threshold <= 1.0:
        raise ValueError(
            f"Invalid face_model_score_threshold {args.face_model_score_threshold}"
        )
    if not 0.0 <= args.lp_model_score_threshold <= 1.0:
        raise ValueError(
            f"Invalid lp_model_score_threshold {args.lp_model_score_threshold}"
        )
    if not 0.0 <= args.nms_iou_threshold <= 1.0:
        raise ValueError(f"Invalid nms_iou_threshold {args.nms_iou_threshold}")
    if not 0 <= args.scale_factor_detections:
        raise ValueError(
            f"Invalid scale_factor_detections {args.scale_factor_detections}"
        )
    if not 1 <= args.output_video_fps or not (
        isinstance(args.output_video_fps, int) and args.output_video_fps % 1 == 0
    ):
        raise ValueError(
            f"Invalid output_video_fps {args.output_video_fps}, should be a positive integer"
        )

    # input/output paths checks
    if args.face_model_path is None and args.lp_model_path is None:
        raise ValueError(
            "Please provide either face_model_path or lp_model_path or both"
        )
    # if args.input_image_folder is None and args.input_video_path is None:
    #     raise ValueError("Please provide either input_image_folder or input_video_path")
    # if args.input_image_folder is not None and args.output_image_folder is None:
    #     raise ValueError(
    #         "Please provide output_image_folder for the visualized image to save."
    #     )
    if args.hammerhead_folder is None or not os.path.exists(args.hammerhead_folder):
        raise ValueError("Please provide a valid hammerhead_folder")
    
    if args.input_video_path is not None and args.output_video_path is None:
        raise ValueError(
            "Please provide output_video_path for the visualized video to save."
        )
    # if args.input_image_folder is not None and not os.path.exists(args.input_image_folder):
    #     raise ValueError(f"{args.input_image_folder} does not exist.")
    if args.input_video_path is not None and not os.path.exists(args.input_video_path):
        raise ValueError(f"{args.input_video_path} does not exist.")
    if args.face_model_path is not None and not os.path.exists(args.face_model_path):
        raise ValueError(f"{args.face_model_path} does not exist.")
    if args.lp_model_path is not None and not os.path.exists(args.lp_model_path):
        raise ValueError(f"{args.lp_model_path} does not exist.")
    # if args.output_image_folder is not None:
    #     print("args.output_image_folder: ", args.output_image_folder)

    # if not os.path.exists(args.output_image_folder):
    #     print("Output folder does not exists")
    # raise NotImplementedError
    # if (args.output_image_folder is not None) and (not os.path.exists(
    #     args.output_image_folder)
    # ):
    #     # raise NotImplementedError
    #     # create_output_directory(args.output_image_folder)
    #     os.makedirs(args.output_image_folder)
    if args.output_video_path is not None and not os.path.exists(
        os.path.dirname(args.output_video_path)
    ):
        create_output_directory(args.output_video_path)

    # check we have write permissions on output paths
    # if args.output_image_folder is not None and not os.access(
    #     args.output_image_folder, os.W_OK
    # ):
    #     raise ValueError(
    #         f"You don't have permissions to write to {args.output_image_folder}. Please grant adequate permissions, or provide a different output path."
    #     )
    if args.output_video_path is not None and not os.access(
        os.path.dirname(args.output_video_path), os.W_OK
    ):
        raise ValueError(
            f"You don't have permissions to write to {args.output_video_path}. Please grant adequate permissions, or provide a different output path."
        )

    return args


@lru_cache
def get_device() -> str:
    """
    Return the device type
    """
    return (
        "cpu"
        if not torch.cuda.is_available()
        else f"cuda:{torch.cuda.current_device()}"
    )


def read_image(image_path: str) -> np.ndarray:
    """
    parameter image_path: absolute path to an image
    Return an image in BGR format
    """
    bgr_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if len(bgr_image.shape) == 2:
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
    return bgr_image


def write_image(image: np.ndarray, image_path: str) -> None:
    """
    parameter image: np.ndarray in BGR format
    parameter image_path: absolute path where we want to save the visualized image
    """
    extension = os.path.splitext(image_path)[1]
    if extension.lower() == ".tiff":
        cv2.imwrite(image_path, image, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
    else:
        cv2.imwrite(image_path, image)


def get_image_tensor(bgr_image: np.ndarray) -> torch.Tensor:
    """
    parameter bgr_image: image on which we want to make detections

    Return the image tensor
    """
    bgr_image_transposed = np.transpose(bgr_image, (2, 0, 1))
    image_tensor = torch.from_numpy(bgr_image_transposed).to(get_device())

    return image_tensor


def get_detections(
    detector: torch.jit._script.RecursiveScriptModule,
    image_tensor: torch.Tensor,
    model_score_threshold: float,
    nms_iou_threshold: float,
) -> List[List[float]]:
    """
    parameter detector: Torchscript module to perform detections
    parameter image_tensor: image tensor on which we want to make detections
    parameter model_score_threshold: model score threshold to filter out low confidence detection
    parameter nms_iou_threshold: NMS iou threshold to filter out low confidence overlapping boxes

    Returns the list of detections
    """
    with torch.no_grad():
        detections = detector(image_tensor)

    boxes, _, scores, _ = detections  # returns boxes, labels, scores, dims

    nms_keep_idx = torchvision.ops.nms(boxes, scores, nms_iou_threshold)
    boxes = boxes[nms_keep_idx]
    scores = scores[nms_keep_idx]

    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()

    score_keep_idx = np.where(scores > model_score_threshold)[0]
    boxes = boxes[score_keep_idx]
    return boxes.tolist()


def scale_box(
    box: List[List[float]], max_width: int, max_height: int, scale: float
) -> List[List[float]]:
    """
    parameter box: detection box in format (x1, y1, x2, y2)
    parameter scale: scaling factor

    Returns a scaled bbox as (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    w = x2 - x1
    h = y2 - y1

    xc = x1 + w / 2
    yc = y1 + h / 2

    w = scale * w
    h = scale * h

    x1 = max(xc - w / 2, 0)
    y1 = max(yc - h / 2, 0)

    x2 = min(xc + w / 2, max_width)
    y2 = min(yc + h / 2, max_height)

    return [x1, y1, x2, y2]


def visualize(
    image: np.ndarray,
    detections: List[List[float]],
    scale_factor_detections: float,
) -> np.ndarray:
    """
    parameter image: image on which we want to make detections
    parameter detections: list of bounding boxes in format [x1, y1, x2, y2]
    parameter scale_factor_detections: scale detections by the given factor to allow blurring more area, 1.15 would mean 15% scaling

    Visualize the input image with the detections and save the output image at the given path
    
    Modification:
    - Instad of creating ellipse around the area, we blur that region
    """
    image_fg = image.copy()
    mask_shape = (image.shape[0], image.shape[1], 1)
    # mask = np.full(mask_shape, 0, dtype=image_fg.dtype)

    for box in detections:
        if scale_factor_detections != 1.0:
            box = scale_box(
                box, image.shape[1], image.shape[0], scale_factor_detections
            )
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        w = x2 - x1
        h = y2 - y1

        # ksize = (image.shape[0] // 2, image.shape[1] // 2)
        ksize = (w//2, h//2)
        # print(ksize)
        image_fg[y1:y2, x1:x2] = cv2.blur(image_fg[y1:y2, x1:x2], ksize)
        # cv2.ellipse(mask, (((x1 + x2) // 2, (y1 + y2) // 2), (w, h), 0), 255, -1)

    # inverse_mask = cv2.bitwise_not(mask)
    # image_bg = cv2.bitwise_and(image, image, mask=inverse_mask)
    # image_fg = cv2.bitwise_and(image_fg, image_fg, mask=mask)
    # image = cv2.add(image_bg, image_fg)

    # return image
    return image_fg

def load_image_files(
    input_image_folder: str
    ) -> List[str]:
    """
    parameter input_image_folder: absolute path to the folder where images are located on which we want to run detector
    """
    file_names = os.listdir(input_image_folder)
    image_files = [f for f in file_names if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg') or f.endswith('.tiff')]

    return image_files

def visualize_image(
    bgr_image: np.ndarray,
    face_detector: torch.jit._script.RecursiveScriptModule,
    lp_detector: torch.jit._script.RecursiveScriptModule,
    face_model_score_threshold: float,
    lp_model_score_threshold: float,
    nms_iou_threshold: float,
)-> List[List[float]]:
    """
    parameter bgr_image: image
    parameter face_detector: face detector model to perform face detections
    parameter lp_detector: face detector model to perform face detections
    parameter face_model_score_threshold: face model score threshold to filter out low confidence detection
    parameter lp_model_score_threshold: license plate model score threshold to filter out low confidence detection
    parameter nms_iou_threshold: NMS iou threshold

    Perform detections on the input image and save the output image at the given path.
    """
    image_tensor = get_image_tensor(bgr_image)
    image_tensor_copy = image_tensor.clone()
    detections = []
    # get face detections
    if face_detector is not None:
        detections.extend(
            get_detections(
                face_detector,
                image_tensor,
                face_model_score_threshold,
                nms_iou_threshold,
            )
        )

    # get license plate detections
    if lp_detector is not None:
        detections.extend(
            get_detections(
                lp_detector,
                image_tensor_copy,
                lp_model_score_threshold,
                nms_iou_threshold,
            )
        )
    # image = visualize(
    #     image,
    #     detections,
    #     scale_factor_detections,
    # )
    # write_image(image, output_image_path)
    return detections


def visualize_video(
    input_video_path: str,
    face_detector: torch.jit._script.RecursiveScriptModule,
    lp_detector: torch.jit._script.RecursiveScriptModule,
    face_model_score_threshold: float,
    lp_model_score_threshold: float,
    nms_iou_threshold: float,
    output_video_path: str,
    scale_factor_detections: float,
    output_video_fps: int,
):
    """
    parameter input_video_path: absolute path to the input video
    parameter face_detector: face detector model to perform face detections
    parameter lp_detector: face detector model to perform face detections
    parameter face_model_score_threshold: face model score threshold to filter out low confidence detection
    parameter lp_model_score_threshold: license plate model score threshold to filter out low confidence detection
    parameter nms_iou_threshold: NMS iou threshold
    parameter output_video_path: absolute path where the visualized video will be saved
    parameter scale_factor_detections: scale detections by the given factor to allow blurring more area
    parameter output_video_fps: fps of the visualized video

    Perform detections on the input video and save the output video at the given path.
    """
    visualized_images = []
    video_reader_clip = VideoFileClip(input_video_path)
    for frame in video_reader_clip.iter_frames():
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        image = frame.copy()
        bgr_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image_tensor = get_image_tensor(bgr_image)
        image_tensor_copy = image_tensor.clone()
        detections = []
        # get face detections
        if face_detector is not None:
            detections.extend(
                get_detections(
                    face_detector,
                    image_tensor,
                    face_model_score_threshold,
                    nms_iou_threshold,
                )
            )
        # get license plate detections
        if lp_detector is not None:
            detections.extend(
                get_detections(
                    lp_detector,
                    image_tensor_copy,
                    lp_model_score_threshold,
                    nms_iou_threshold,
                )
            )
        visualized_images.append(
            visualize(
                image,
                detections,
                scale_factor_detections,
            )
        )

    video_reader_clip.close()

    if visualized_images:
        video_writer_clip = ImageSequenceClip(visualized_images, fps=output_video_fps)
        video_writer_clip.write_videofile(output_video_path)
        video_writer_clip.close()


if __name__ == "__main__":
    args = validate_inputs(parse_args())
    if args.face_model_path is not None:
        face_detector = torch.jit.load(args.face_model_path, map_location="cpu").to(
            get_device()
        )
        face_detector.eval()
    else:
        face_detector = None

    if args.lp_model_path is not None:
        lp_detector = torch.jit.load(args.lp_model_path, map_location="cpu").to(
            get_device()
        )
        lp_detector.eval()
    else:
        lp_detector = None

    if args.hammerhead_folder is not None:
        print("Bluring LP in topbots...")
        input_topbot_folder = os.path.join(args.hammerhead_folder, 'topbot-raw')
        output_topbot_folder = os.path.join(args.hammerhead_folder, 'topbot')

        if not os.path.exists(output_topbot_folder):
            os.makedirs(output_topbot_folder)
            image_files = load_image_files(args.input_topbot_folder)
            for img_file in tqdm.tqdm(sorted(image_files)):
                input_image_path = os.path.join(args.input_image_folder, img_file)
                output_image_path = os.path.join(args.output_image_folder, img_file)

                bgr_image = read_image(input_image_path)
                image = bgr_image.copy()

                detections = visualize_image(
                    bgr_image,
                    face_detector,
                    lp_detector,
                    args.face_model_score_threshold,
                    args.lp_model_score_threshold,
                    args.nms_iou_threshold,
                )
                
                image = visualize(
                    image,
                    detections,
                    args.scale_factor_detections,
                )
                write_image(image, output_image_path)
        print("Finished bluring LP in topbots.")
        print("Bluring LP in left-rect and depth-colormap...")
        input_topbot_folder_1 = os.path.join(args.hammerhead_folder, 'left-rect-raw')
        output_topbot_folder_1 = os.path.join(args.hammerhead_folder, 'left-rect')

        input_topbot_folder_2 = os.path.join(args.hammerhead_folder, 'depth-colormap-raw')
        output_topbot_folder_2 = os.path.join(args.hammerhead_folder, 'depth-colormap')
        if not os.path.exists(output_topbot_folder_1) or not os.path.exists(output_topbot_folder_2):
            os.makedirs(output_topbot_folder_1)
            os.makedirs(output_topbot_folder_2)
            image_files_1 = load_image_files(input_topbot_folder_1)
            image_files_2 = load_image_files(input_topbot_folder_2)
            print("Number of images: ", len(image_files_1))

            for img_file in tqdm.tqdm(sorted(image_files_1)):
                # run detection on left-rect
                input_image_path = os.path.join(input_topbot_folder_1, img_file)
                output_image_path = os.path.join(output_topbot_folder_1, img_file)

                bgr_image = read_image(input_image_path)
                image = bgr_image.copy()

                detections = visualize_image(
                    bgr_image,
                    face_detector,
                    lp_detector,
                    args.face_model_score_threshold,
                    args.lp_model_score_threshold,
                    args.nms_iou_threshold,
                )
                
                image = visualize(
                    image,
                    detections,
                    args.scale_factor_detections,
                )
                write_image(image, output_image_path)

                # use detection on left-rect to blur depth-colored
                input_image_path = os.path.join(input_topbot_folder_2, img_file)
                output_image_path = os.path.join(output_topbot_folder_2, img_file)

                bgr_image = read_image(input_image_path)
                image = bgr_image.copy()
                image = visualize(
                    image,
                    detections,
                    args.scale_factor_detections,
                )
                write_image(image, output_image_path)


    if args.input_video_path is not None:
        raise NotImplementedError("Video visualization is not implemented yet.")
        visualize_video(
            args.input_video_path,
            face_detector,
            lp_detector,
            args.face_model_score_threshold,
            args.lp_model_score_threshold,
            args.nms_iou_threshold,
            args.output_video_path,
            args.scale_factor_detections,
            args.output_video_fps,
        )
