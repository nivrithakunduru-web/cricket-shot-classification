
import cv2
import numpy as np
import torch
import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)


class BiGRUWithAttention(nn.Module):
    def __init__(
        self,
        input_size=1280,
        hidden_size=256,
        num_layers=2,
        num_classes=10,
        dropout=0.4
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.attention = nn.Linear(hidden_size * 2, 1)
        self.classifier = nn.Linear(
            hidden_size * 2, num_classes
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        scores = self.attention(gru_out).squeeze(-1)
        weights = torch.softmax(scores, dim=1)

        context = torch.sum(
            gru_out * weights.unsqueeze(-1),
            dim=1
        )

        return self.classifier(context)


class CricketShotPredictor:
    def __init__(self, checkpoint_path, device=None):
        self.device = torch.device(
            device or (
                "cuda" if torch.cuda.is_available()
                else "cpu"
            )
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        self.class_names = checkpoint["class_names"]
        self.config = checkpoint["preprocessing"]

        self.model = BiGRUWithAttention(
            **checkpoint["model_config"]
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )

        self.model.to(self.device).eval()

        # Use the same pretrained EfficientNet-B0
        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = efficientnet_b0(weights=weights)
        self.backbone.classifier = nn.Identity()
        self.backbone.to(self.device).eval()

    def preprocess_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError("Cannot open video")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise ValueError("Video has no readable frames")

        num_frames = self.config["num_frames"]
        image_size = self.config["image_size"]

        indices = np.linspace(
            0, total - 1, num_frames
        ).astype(int)

        mean = np.array(
            self.config["mean"], dtype=np.float32
        )
        std = np.array(
            self.config["std"], dtype=np.float32
        )

        frames = []

        try:
            for idx in indices:
                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    int(idx)
                )
                ok, frame = cap.read()

                if not ok:
                    raise ValueError(
                        f"Failed to read frame {idx}"
                    )

                frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                frame = cv2.resize(
                    frame, (image_size, image_size)
                )

                frame = frame.astype(
                    np.float32
                ) / 255.0

                frame = (frame - mean) / std
                frame = np.transpose(frame, (2, 0, 1))
                frames.append(frame)

        finally:
            cap.release()

        return torch.from_numpy(
            np.stack(frames)
        ).float()

    @torch.inference_mode()
    def predict(self, video_path):
        frames = self.preprocess_video(
            video_path
        ).to(self.device)

        # Process frames in small batches to
        # avoid GPU/CPU memory spikes.
        chunks = []

        for chunk in frames.split(8):
            chunks.append(
                self.backbone(chunk)
            )

        features = torch.cat(
            chunks, dim=0
        ).unsqueeze(0)

        logits = self.model(features)
        probabilities = torch.softmax(
            logits, dim=1
        )[0]

        confidence, index = probabilities.max(dim=0)

        return {
            "predicted_class": self.class_names[
                index.item()
            ],
            "confidence": confidence.item() * 100,
            "probabilities": {
                name: round(prob.item() * 100, 2)
                for name, prob in zip(
                    self.class_names, probabilities
                )
            }
        }
