import io
import os
import traceback
import uuid
from pathlib import Path
from typing import Any

import cv2
import supervision as sv

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torchvision
import numpy as np
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

from .env_config import load_env_file
from .path_manager import create_all_paths
from .path_manager import get_video_folder_path
from .path_manager import get_background_temp_image_folder
from .path_manager import get_foreground_temp_image_folder
from .path_manager import get_images_path
from .path_manager import get_source_images_path
from .path_manager import get_upload_path
from .path_manager import get_checkpoint_path
from .path_manager import get_config_path
from .path_manager import get_temp_file_path
from .path_manager import get_frame_path
from .path_manager import get_source_frame_path
from .path_manager import get_masked_video_path
from .path_manager import get_preview_mask_frame_name

load_env_file()

MAX_PROCESSING_FPS = float(os.environ.get("FREEZE_ME_MAX_FPS", "15"))
SAM_DEVICE = os.environ.get("SAM_DEVICE", "auto").lower()
SAM_IMAGE_SIZE = int(os.environ.get("SAM_IMAGE_SIZE", "512"))


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


SAM2_ASYNC_LOADING = _env_flag("FREEZE_ME_SAM2_ASYNC_LOADING", True)
SAM2_MULTIMASK = _env_flag("FREEZE_ME_SAM2_MULTIMASK", False)
WRITE_SEGMENTATION_CUTOUTS = _env_flag("FREEZE_ME_WRITE_SEGMENTATION_CUTOUTS", True)


def _resolve_device() -> torch.device:
    if SAM_DEVICE == "cpu":
        return torch.device("cpu")
    if SAM_DEVICE == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("SAM_DEVICE is set to 'cuda', but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


device = _resolve_device()
predictor: SAM2VideoPredictor | None = None

colors = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
mask_annotator = sv.MaskAnnotator(
    color=sv.ColorPalette.from_hex(colors),
    color_lookup=sv.ColorLookup.TRACK)

import ffmpeg
from fastapi import UploadFile
import shutil

inference_state: {}
fps = 0
segmentation_sessions: dict[str, dict[str, Any]] = {}


def _patch_predictor_dtype_handling(predictor_instance: SAM2VideoPredictor) -> None:
    original_run_single_frame_inference = predictor_instance._run_single_frame_inference
    original_run_memory_encoder = predictor_instance._run_memory_encoder

    def run_single_frame_inference_fp32(*args, **kwargs):
        compact_current_out, pred_masks_gpu = original_run_single_frame_inference(*args, **kwargs)
        maskmem_features = compact_current_out.get("maskmem_features")
        if isinstance(maskmem_features, torch.Tensor) and maskmem_features.dtype != torch.float32:
            compact_current_out["maskmem_features"] = maskmem_features.to(torch.float32)
        return compact_current_out, pred_masks_gpu

    def run_memory_encoder_fp32(*args, **kwargs):
        maskmem_features, maskmem_pos_enc = original_run_memory_encoder(*args, **kwargs)
        if isinstance(maskmem_features, torch.Tensor) and maskmem_features.dtype != torch.float32:
            maskmem_features = maskmem_features.to(torch.float32)
        return maskmem_features, maskmem_pos_enc

    predictor_instance._run_single_frame_inference = run_single_frame_inference_fp32
    predictor_instance._run_memory_encoder = run_memory_encoder_fp32


def _parse_fps(fps_string: str) -> float:
    numerator, denominator = fps_string.split("/")
    denominator_value = float(denominator)
    if denominator_value == 0:
        return 0.0
    return round(float(numerator) / denominator_value, 2)


def _probe_video(video_path: Path) -> tuple[dict[str, Any], dict[str, Any], float]:
    details = ffmpeg.probe(video_path.as_posix(), cmd="ffprobe")
    video_stream = next(stream for stream in details["streams"] if stream["codec_type"] == "video")
    source_fps = _parse_fps(video_stream["r_frame_rate"])
    return details, video_stream, source_fps


def _get_processing_fps(source_fps: float) -> float:
    if MAX_PROCESSING_FPS <= 0:
        return source_fps
    if source_fps <= 0:
        return MAX_PROCESSING_FPS
    return min(source_fps, MAX_PROCESSING_FPS)


def _remove_matching_files(folder: Path, pattern: str) -> None:
    if not folder.exists():
        return
    for file_path in folder.glob(pattern):
        if file_path.is_file():
            file_path.unlink(missing_ok=True)


def _extract_video_frames(
    video_path: Path,
    image_folder: Path,
    processing_fps: float,
    source_fps: float,
    max_dimension: int | None = None,
) -> None:
    image_folder.mkdir(parents=True, exist_ok=True)
    _remove_matching_files(image_folder, "*.jpeg")

    stream = ffmpeg.input(video_path.as_posix())
    if source_fps > processing_fps + 0.01:
        stream = stream.filter("fps", fps=processing_fps)
    if max_dimension is not None and max_dimension > 0:
        stream = stream.filter(
            "scale",
            f"if(gt(iw,ih),{max_dimension},-2)",
            f"if(gt(iw,ih),-2,{max_dimension})",
        )

    (
        stream.output(image_folder.joinpath("%05d.jpeg").as_posix(), start_number=0, **{"q:v": "2"})
        .overwrite_output()
        .run(quiet=True)
    )


def _get_frame_paths(video_id: str) -> list[Path]:
    image_path = get_images_path(video_id)
    return sorted(sv.list_files_with_extensions(directory=image_path.__str__(), extensions=["jpeg"]))


def _get_source_frame_paths(video_id: str) -> list[Path]:
    image_path = get_source_images_path(video_id)
    return sorted(sv.list_files_with_extensions(directory=image_path.__str__(), extensions=["jpeg"]))


def clear_segmentation_session(video_id: str) -> None:
    segmentation_sessions.pop(video_id, None)
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _clear_generated_artifacts(video_id: str) -> None:
    image_folder = get_images_path(video_id)
    source_image_folder = get_source_images_path(video_id)
    preview_folder = get_preview_mask_frame_name(video_id, 0).parent
    foreground_folder = get_foreground_temp_image_folder(video_id)
    background_folder = get_background_temp_image_folder(video_id)

    for folder, pattern in (
        (image_folder, "*.jpeg"),
        (source_image_folder, "*.jpeg"),
        (preview_folder, "*.png"),
        (foreground_folder, "*.png"),
        (background_folder, "*.png"),
    ):
        _remove_matching_files(folder, pattern)

    for file_path in (
        get_masked_video_path(video_id),
        get_temp_file_path(video_id),
    ):
        file_path.unlink(missing_ok=True)


def get_predictor() -> SAM2VideoPredictor:
    global predictor

    if predictor is not None:
        return predictor

    checkpoint_path = Path(get_checkpoint_path())
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found at {checkpoint_path}. "
            "Run `python download_checkpoint.py` in the backend folder first."
        )

    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Using checkpoint:", checkpoint_path)
    print("Using config:", get_config_path())
    print("Using device:", device)
    print("Using SAM image size:", SAM_IMAGE_SIZE)
    print("Using SAM multimask:", SAM2_MULTIMASK)

    if device.type == "cuda":
        # Do not force a global autocast context for SAM2.
        # On some local CUDA setups this mixes BF16 activations with FP32 weights
        # and crashes during propagation with dtype mismatch errors.
        torch.backends.cudnn.benchmark = True
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    hydra_overrides = [f"++model.image_size={SAM_IMAGE_SIZE}"]
    if not SAM2_MULTIMASK:
        hydra_overrides.extend(
            [
                "++model.multimask_output_in_sam=false",
                "++model.multimask_output_for_tracking=false",
            ]
        )

    predictor = build_sam2_video_predictor(
        get_config_path(),
        checkpoint_path.as_posix(),
        device=device,
        hydra_overrides_extra=hydra_overrides,
    )
    _patch_predictor_dtype_handling(predictor)
    return predictor


def _get_segmentation_session(video_id: str) -> dict[str, Any]:
    session = segmentation_sessions.get(video_id)
    if session is None:
        raise RuntimeError(
            f"Segmentation state for video '{video_id}' is not initialized. "
            f"Call initialize_segmentation first."
        )
    return session


async def _ensure_segmentation_session(video_id: str) -> dict[str, Any]:
    session = segmentation_sessions.get(video_id)
    if session is not None:
        return session

    print(f"Segmentation session for '{video_id}' was missing; reinitializing it.")
    await initialize_segmentation(video_id)
    return _get_segmentation_session(video_id)


def _normalize_tensor_dtypes(value):
    if isinstance(value, torch.Tensor) and value.is_floating_point() and value.dtype != torch.float32:
        return value.to(torch.float32)
    if isinstance(value, dict):
        return {key: _normalize_tensor_dtypes(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_tensor_dtypes(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_tensor_dtypes(item) for item in value)
    return value


def _normalize_inference_state(session: dict[str, Any]) -> None:
    inference_state = session.get("inference_state")
    if inference_state is None:
        return

    for key in ("output_dict_per_obj", "temp_output_dict_per_obj", "cached_features", "constants"):
        if key in inference_state:
            inference_state[key] = _normalize_tensor_dtypes(inference_state[key])


def _extract_masks(mask_logits):
    mask_data = (mask_logits > 0.0).cpu().numpy()
    n, x, h, w = mask_data.shape
    masks = mask_data.reshape(n * x, h, w)
    combined_mask = np.any(masks, axis=0) if masks.shape[0] > 0 else np.zeros((h, w), dtype=bool)
    return masks, combined_mask


def _build_detections(masks, out_obj_ids):
    if masks.shape[0] == 0:
        return None
    return sv.Detections(
        xyxy=sv.mask_to_xyxy(masks=masks),
        mask=masks,
        tracker_id=np.array(out_obj_ids)
    )


def _scale_points_to_frame(points, from_shape, to_shape):
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)

    from_height, from_width = from_shape[:2]
    to_height, to_width = to_shape[:2]
    scale_x = to_width / from_width
    scale_y = to_height / from_height

    scaled_points = np.array(points, dtype=np.float32).copy()
    scaled_points[:, 0] *= scale_x
    scaled_points[:, 1] *= scale_y
    return scaled_points


def _resize_masks_to_frame(masks, combined_mask, frame_shape):
    target_height, target_width = frame_shape[:2]
    if combined_mask.shape[:2] == (target_height, target_width):
        return masks, combined_mask

    resized_combined_mask = cv2.resize(
        combined_mask.astype(np.uint8),
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    if masks.shape[0] == 0:
        return masks, resized_combined_mask

    resized_masks = np.stack(
        [
            cv2.resize(
                mask.astype(np.uint8),
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            for mask in masks
        ],
        axis=0,
    )
    return resized_masks, resized_combined_mask


def _write_cutout_frames(frame, combined_mask, foreground_path: Path, background_path: Path) -> None:
    frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    alpha_mask = combined_mask.astype(np.uint8) * 255

    foreground = frame_bgra.copy()
    foreground[:, :, 3] = alpha_mask
    if not cv2.imwrite(str(foreground_path), foreground):
        raise IOError(f"Failed to write foreground cutout to {foreground_path}.")

    frame_bgra[:, :, 3] = 255 - alpha_mask
    if not cv2.imwrite(str(background_path), frame_bgra):
        raise IOError(f"Failed to write background cutout to {background_path}.")


async def save_video(file: UploadFile):
    video_id = uuid.uuid4().hex.__str__() + Path(file.filename).suffix
    create_all_paths(video_id)
    path = get_upload_path(video_id)
    video_data = io.BytesIO(await file.read())
    with open(path, "wb") as f:
        f.write(video_data.getbuffer())

    _, _, source_fps = _probe_video(path)
    processing_fps = _get_processing_fps(source_fps)
    _extract_video_frames(path, get_images_path(video_id), processing_fps, source_fps, max_dimension=SAM_IMAGE_SIZE)
    _extract_video_frames(path, get_source_images_path(video_id), processing_fps, source_fps)

    return video_id


async def get_video_details(video_id):
    try:
        path = get_upload_path(video_id)
        details, _, source_fps = _probe_video(path)
        global fps
        total_frames = len(_get_source_frame_paths(video_id))
        details["total_frames"] = total_frames
        details["processing_fps"] = _get_processing_fps(source_fps)
        details["source_fps"] = source_fps
        fps = details["processing_fps"]
        print("FPS: ", fps)
        print("Total frames: ", total_frames)
        return details
    except Exception as e:
        print(e)
        print(e.__traceback__)
        print(traceback.format_exc())

        return ""


async def initialize_segmentation(video_id):
    try:
        sam_frame_paths = _get_frame_paths(video_id)
        source_frame_paths = _get_source_frame_paths(video_id)
        total_frames = len(sam_frame_paths)
        if total_frames == 0:
            raise FileNotFoundError(f"No extracted frames found for video '{video_id}'.")
        if len(source_frame_paths) != total_frames:
            raise RuntimeError(
                f"Expected matching SAM and source frame counts for video '{video_id}', "
                f"got {total_frames} and {len(source_frame_paths)}."
            )

        predictor_instance = get_predictor()
        _, _, source_fps = _probe_video(get_upload_path(video_id))
        sam_reference_frame = cv2.imread(str(sam_frame_paths[0]), cv2.IMREAD_UNCHANGED)
        source_reference_frame = cv2.imread(str(source_frame_paths[0]), cv2.IMREAD_UNCHANGED)
        if sam_reference_frame is None or source_reference_frame is None:
            raise FileNotFoundError(f"Could not load reference frames for video '{video_id}'.")
        segmentation_sessions[video_id] = {
            "inference_state": predictor_instance.init_state(
                video_path=get_images_path(video_id).__str__(),
                offload_video_to_cpu=False,
                offload_state_to_cpu=False,
                async_loading_frames=SAM2_ASYNC_LOADING,
            ),
            "processing_fps": _get_processing_fps(source_fps),
            "frame_paths": sam_frame_paths,
            "source_frame_paths": source_frame_paths,
            "sam_frame_shape": sam_reference_frame.shape[:2],
            "source_frame_shape": source_reference_frame.shape[:2],
            "points": [[] for _ in range(total_frames)],
            "labels": [[] for _ in range(total_frames)],
            "segmentation_dirty": True,
        }
        _normalize_inference_state(segmentation_sessions[video_id])
    except Exception as e:
        segmentation_sessions.pop(video_id, None)
        print(e)
        print(e.__traceback__)
        print(traceback.format_exc())
        raise


async def get_frame(video_id, frame_id):
    try:
        return get_source_frame_path(video_id, frame_id)
    except Exception as e:
        print(e)
        print(e.__traceback__)
        print(traceback.format_exc())


async def add_new_point_to_segmentation(video_id, point_x, point_y, point_type, frame_num):
    try:
        predictor_instance = get_predictor()
        session = await _ensure_segmentation_session(video_id)
        frame_num = int(frame_num)
        if frame_num < 0 or frame_num >= len(session["points"]):
            raise IndexError(f"Frame index {frame_num} is out of range for video '{video_id}'.")

        session["points"][frame_num].append([float(point_x), float(point_y)])
        session["labels"][frame_num].append(int(point_type))
        scaled_points = _scale_points_to_frame(
            session["points"][frame_num],
            session["source_frame_shape"],
            session["sam_frame_shape"],
        )

        _, out_obj_ids, out_mask_logits = predictor_instance.add_new_points_or_box(
            inference_state=session["inference_state"],
            frame_idx=frame_num,
            obj_id=1,
            points=scaled_points,
            labels=np.array(session["labels"][frame_num], dtype=np.int32),
        )
        session["segmentation_dirty"] = True
        _normalize_inference_state(session)

        masks, combined_mask = _extract_masks(out_mask_logits)
        source_frame_path = get_source_frame_path(video_id, frame_num)
        frame = cv2.imread(source_frame_path.__str__())
        if frame is None:
            raise FileNotFoundError(f"Could not load frame {source_frame_path}.")

        masks, combined_mask = _resize_masks_to_frame(masks, combined_mask, frame.shape)
        detections = _build_detections(masks, out_obj_ids)

        if detections is not None:
            frame = mask_annotator.annotate(frame, detections)
        preview_path = get_preview_mask_frame_name(video_id, frame_num)
        if not cv2.imwrite(preview_path.__str__(), frame):
            raise IOError(f"Failed to write preview mask frame to {preview_path}.")

        return preview_path
    except Exception as e:
        print(e)
        print(e.__traceback__)
        print(traceback.format_exc())
        raise


async def get_masked_video(video_id):
    try:
        predictor_instance = get_predictor()
        session = await _ensure_segmentation_session(video_id)
        _normalize_inference_state(session)
        output_path = get_masked_video_path(video_id)
        if not session.get("segmentation_dirty", True) and output_path.exists():
            return output_path.__str__()

        video_path = get_upload_path(video_id)
        source_video_info = sv.VideoInfo.from_video_path(video_path.__str__())
        frames_paths = session.get("frame_paths") or _get_frame_paths(video_id)
        source_frame_paths = session.get("source_frame_paths") or _get_source_frame_paths(video_id)
        session["frame_paths"] = frames_paths
        session["source_frame_paths"] = source_frame_paths
        video_info = sv.VideoInfo(
            width=source_video_info.width,
            height=source_video_info.height,
            fps=session.get("processing_fps", source_video_info.fps),
            total_frames=len(source_frame_paths),
        )
        temp_file = get_temp_file_path(video_id)
        temp_file.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)

        foreground_folder = get_foreground_temp_image_folder(video_id)
        background_folder = get_background_temp_image_folder(video_id)
        if WRITE_SEGMENTATION_CUTOUTS:
            _remove_matching_files(foreground_folder, "*.png")
            _remove_matching_files(background_folder, "*.png")

        with sv.VideoSink(temp_file.__str__(), video_info=video_info) as sink:
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor_instance.propagate_in_video(
                session["inference_state"],
                start_frame_idx=0,
            ):
                frame = cv2.imread(str(source_frame_paths[out_frame_idx]))
                if frame is None:
                    raise FileNotFoundError(f"Could not load frame {source_frame_paths[out_frame_idx]}.")

                masks, combined_mask = _extract_masks(out_mask_logits)
                masks, combined_mask = _resize_masks_to_frame(masks, combined_mask, frame.shape)
                detections = _build_detections(masks, out_obj_ids)
                base_path = Path(os.path.basename(source_frame_paths[out_frame_idx])).stem

                if WRITE_SEGMENTATION_CUTOUTS:
                    _write_cutout_frames(
                        frame,
                        combined_mask,
                        foreground_folder.joinpath(base_path + ".png"),
                        background_folder.joinpath(base_path + ".png"),
                    )

                if detections is not None:
                    frame = mask_annotator.annotate(frame, detections)
                sink.write_frame(frame)
                del frame, masks
        (ffmpeg
         .input(temp_file.__str__())
         .output(
            output_path.__str__(),
            vcodec='libx264',
            movflags='faststart',
            an=None
        )
         .overwrite_output().run(quiet=True)
         )
        temp_file.unlink(missing_ok=True)
        session["segmentation_dirty"] = False
        print(output_path.__str__())
        return output_path.__str__()
    except Exception as e:
        print(e)
        print(e.__traceback__)
        print(traceback.format_exc())
        raise


async def cut_video(video_id: str, start_time: float, end_time: float):
    try:
        # Pfade initialisieren
        video_folder = get_video_folder_path(video_id)
        original_video_path = get_upload_path(video_id)
        temp_video_path = video_folder.joinpath(f"temp_{video_id}")

        # Überprüfen, ob das Verzeichnis existiert
        if not video_folder.exists():
            raise FileNotFoundError(f"Verzeichnis {video_folder.__str__()} existiert nicht.")

        # Überprüfen, ob das Originalvideo existiert
        if not original_video_path.exists():
            raise FileNotFoundError(f"Originalvideo {original_video_path.__str__()} wurde nicht gefunden.")

        _, _, source_fps = _probe_video(original_video_path)
        processing_fps = _get_processing_fps(source_fps)

        global fps
        fps = processing_fps
        print(f"Start Time: {start_time}")
        print(f"End Time: {end_time}")
        print(f"Source FPS: {source_fps}")
        print(f"Processing FPS: {processing_fps}")

        input_file = ffmpeg.input(original_video_path.__str__())
        ffmpeg.output(
            input_file.trim(start=start_time, end=end_time).setpts('PTS-STARTPTS'),
            temp_video_path.__str__(),
            vcodec='libx264',
            movflags='faststart',
            an=None
        ).overwrite_output().run(quiet=True)

        # Überprüfen, ob die temporäre Datei erfolgreich erstellt wurde
        if temp_video_path.exists():
            # Speichern unter einer neuen zufälligen ID
            new_video_file_name = await save_cut_video(temp_video_path, video_id)
            temp_video_path.unlink()  # Temporäre Datei löschen
        else:
            raise Exception(f"Das temporäre Video {temp_video_path} wurde nicht erfolgreich erstellt.")
        clear_segmentation_session(video_id)
        _clear_generated_artifacts(video_id)
        _extract_video_frames(original_video_path, get_images_path(video_id), processing_fps, source_fps, max_dimension=SAM_IMAGE_SIZE)
        _extract_video_frames(original_video_path, get_source_images_path(video_id), processing_fps, source_fps)
        return new_video_file_name

    except Exception as e:
        print(f"Fehler beim Schneiden des Videos: {e}")
        raise e


async def save_cut_video(file_path: Path, video_id: str):
    path = get_upload_path(video_id)

    shutil.copy(file_path, path)
    return path
