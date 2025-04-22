from ultralytics import YOLO
from typing import List, Union, Tuple, Dict, Optional
from PIL import Image
import numpy as np
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Class to handle Detector results"""
    frame_path: str
    track_id: Optional[int]
    box: List[float]
    score: float
    class_id: int

class Detector(YOLO):
    """
    Optimized YOLO-based detector with tracking support.
    Handles batch processing and parallel operations efficiently.
    """

    def __init__(self, model_path: str, mode: str = "detect"):
        """
        Initialize the Detector by loading a YOLO model.

        Args:
            model_path (str): Path to the YOLO model file (.pt or .onnx).
            mode (str): Task type ('detect', 'track').

        Raises:
            ValueError: If mode is not supported.
            RuntimeError: If the model fails to load.
        """
        supported_modes = {"detect", "track"}
        mode = mode.lower()

        if mode not in supported_modes:
            raise ValueError(
                f"Unsupported mode: '{mode}'. Supported modes are: {supported_modes}"
            )

        try:
            super().__init__(model=model_path)
            self.model_path = model_path
            self.mode = mode

            self.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from '{model_path}': {e}")

    def __repr__(self):
        return f"Detector(mode={self.mode}, model={self.model_path})"

    def process_batch(
        self, 
        sources: List[Union[str, np.ndarray]], 
        batch_size: int = 8,
        max_workers: int = 4,
        **kwargs
    ) -> List[DetectionResult]:
        """
        Process a batch of images in parallel with optimized memory usage.

        Args:
            sources: List of image paths or arrays.
            batch_size: Number of images to process in each sub-batch.
            max_workers: Maximum number of parallel workers.
            **kwargs: Extra args for YOLO predict/track.

        Returns:
            List of detection results.
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(0, len(sources), batch_size):
                batch = sources[i:i + batch_size]
                futures.append(executor.submit(self._process_single_batch, batch, **kwargs))
            
            for future in as_completed(futures):
                try:
                    results.extend(future.result())
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
        
        return results

    def _process_single_batch(
        self, 
        batch: List[Union[str, np.ndarray]], 
        **kwargs
    ) -> List[DetectionResult]:
        """Process a single batch of images."""
        batch_results = []
        start_time = time.time()
        
        yolo_results = self.track(batch, **kwargs) if self.mode == "track" else self.predict(batch, **kwargs)
        
        for idx, result in enumerate(yolo_results):
            detections = result.boxes
            frame_name = batch[idx] if isinstance(batch[idx], str) else f"frame_{idx}"

            if detections is None or detections.xyxy is None:
                continue

            xyxy = detections.xyxy.cpu().numpy()
            confs = detections.conf.cpu().numpy()
            classes = detections.cls.cpu().numpy().astype(int)
            ids = (
                detections.id.cpu().numpy().astype(int)
                if detections.id is not None
                else np.full(len(xyxy), -1)
            )

            for i in range(len(xyxy)):
                batch_results.append(DetectionResult(
                    frame_path=str(frame_name),
                    track_id=int(ids[i]) if ids[i] != -1 else None,
                    box=xyxy[i].tolist(),
                    score=float(confs[i]),
                    class_id=int(classes[i])
                ))
        
        logger.debug(f"Processed batch of {len(batch)} in {time.time() - start_time:.2f}s")
        return batch_results

class Classifier(YOLO):
    """
    Optimized YOLO-based classifier with batch processing support.
    """
    def __init__(self, model_path: str, feature_name: str):
        """
        Initialize the Classifier.

        Args:
            model_path: Path to the YOLO model file.
            feature_name: Feature name for classification results.
        """
        super().__init__(model=model_path)
        self.feature_name = feature_name
        self.predict(np.zeros((224, 224, 3), dtype=np.uint8), verbose=False)

    def process_batch(
        self, 
        predictions: List[dict], 
        max_workers: int = 4
    ) -> List[dict]:
        """
        Process a batch of predictions in parallel.

        Args:
            predictions: List of detection dictionaries.
            max_workers: Maximum parallel workers.

        Returns:
            List of enhanced predictions with classification results.
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            process_fn = partial(self._process_single_prediction)
            return list(executor.map(process_fn, predictions))

    def _process_single_prediction(self, prediction: dict) -> dict:
        """
        Process a single prediction by classifying the detected region.
        """
        if "box" not in prediction or "frame_path" not in prediction:
            raise ValueError("Prediction must contain 'box' and 'frame_path'")

        try:
            xmin, ymin, xmax, ymax = map(int, prediction["box"])
            with Image.open(prediction["frame_path"]) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                detection = img.crop((xmin, ymin, xmax, ymax))

            result = self.predict(detection, verbose=False)[0].probs
            prediction[f"{self.feature_name}_class"] = result.top1
            prediction[f"{self.feature_name}_confidence"] = float(result.top1conf)
            
            return prediction
        except Exception as e:
            logger.error(f"Error processing {prediction.get('frame_path')}: {e}")
            prediction[f"{self.feature_name}_class"] = -1
            prediction[f"{self.feature_name}_confidence"] = 0.0
            return prediction
