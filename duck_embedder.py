"""
duck_embedder.py -- deep appearance embeddings for duck re-identification.

Turns a duck crop into a unit-norm feature vector so two crops of the SAME
duck score high (cosine) and different ducks score low -- far more
discriminative than a colour histogram between similar-coloured rubber ducks.

Backends, tried in this order (first that imports wins):
  1. SigLIP  (transformers)  -- best; strong general visual features.
  2. ResNet50 (torchvision)  -- solid fallback, penultimate-layer features.
  3. None                    -- returns None; caller falls back to colour hist.

Only 18 vectors are ever compared, so there is NO vector DB / FAISS here --
a plain numpy cosine over a handful of rows is microseconds. Keep it simple.

Usage:
    emb = DuckEmbedder(device="cuda:0")   # loads once
    v = emb.embed(crop_bgr)               # -> np.ndarray (unit norm) or None
    sim = DuckEmbedder.cosine(v1, v2)     # -> float in [-1, 1]
"""

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


class DuckEmbedder:
    def __init__(self, device="cpu", backend="auto", verbose=True):
        self.device = device
        self.backend = None
        self._model = None
        self._processor = None
        self._torch = None
        self._preprocess = None
        self.dim = None

        order = (["siglip", "resnet"] if backend == "auto" else [backend])
        for b in order:
            try:
                if b == "siglip" and self._init_siglip():
                    self.backend = "siglip"; break
                if b == "resnet" and self._init_resnet():
                    self.backend = "resnet"; break
            except Exception as e:
                if verbose:
                    print(f"[EMB] backend '{b}' unavailable: {e}")
        if verbose:
            if self.backend:
                print(f"[EMB] using backend '{self.backend}' (dim={self.dim}) "
                      f"on {self.device}")
            else:
                print("[EMB] no embedding backend -> falling back to colour hist")

    # ------------------------------------------------------------------ #
    #  backend init                                                      #
    # ------------------------------------------------------------------ #
    def _init_siglip(self):
        import torch
        from transformers import AutoModel, AutoProcessor
        self._torch = torch
        name = "google/siglip-base-patch16-224"
        self._processor = AutoProcessor.from_pretrained(name)
        self._model = AutoModel.from_pretrained(name).to(self.device).eval()
        # probe output dim with a dummy
        self.dim = int(self._model.config.vision_config.hidden_size)
        return True

    def _init_resnet(self):
        import torch
        import torchvision
        from torchvision import transforms
        self._torch = torch
        m = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT)
        m.fc = torch.nn.Identity()   # take 2048-d penultimate features
        self._model = m.to(self.device).eval()
        self._preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.dim = 2048
        return True

    # ------------------------------------------------------------------ #
    #  embed                                                             #
    # ------------------------------------------------------------------ #
    def embed(self, crop_bgr):
        """crop_bgr: HxWx3 BGR uint8 (OpenCV). Returns unit-norm np.ndarray
        or None if no backend / bad crop."""
        if self.backend is None or crop_bgr is None or crop_bgr.size == 0:
            return None
        try:
            rgb = crop_bgr[:, :, ::-1]  # BGR -> RGB
            if self.backend == "siglip":
                v = self._embed_siglip(rgb)
            else:
                v = self._embed_resnet(rgb)
            if v is None:
                return None
            n = np.linalg.norm(v)
            return (v / n) if n > 0 else None
        except Exception:
            return None

    def _embed_siglip(self, rgb):
        torch = self._torch
        inputs = self._processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        return feats.squeeze(0).float().cpu().numpy()

    def _embed_resnet(self, rgb):
        torch = self._torch
        x = self._preprocess(np.ascontiguousarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self._model(x)
        return feats.squeeze(0).float().cpu().numpy()

    # ------------------------------------------------------------------ #
    #  similarity                                                        #
    # ------------------------------------------------------------------ #
    @staticmethod
    def cosine(a, b):
        if a is None or b is None:
            return None
        # both are unit norm already, but be safe
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))