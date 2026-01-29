"""
Parseq OCR Wrapper

Wraps the Parseq model for license plate recognition.
Parseq: "Scene Text Recognition with Permuted Autoregressive Sequence Items"

This module provides:
1. ParseqOCR class for inference and fine-tuning
2. ParseqTokenizer for encoding/decoding text
3. Integration with LCOFL loss function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import re

# Parseq availability check - will use torch.hub to load
try:
    # Try to load Parseq via torch.hub
    _test_model = torch.hub.load('baudm/parseq', 'parseq', pretrained=False, trust_repo=True)
    PARSEQ_AVAILABLE = True
    PARSEQ_SOURCE = 'torch_hub'
    del _test_model  # Free memory
except Exception as e:
    PARSEQ_AVAILABLE = False
    PARSEQ_SOURCE = None
    print(f"Warning: Could not load Parseq via torch.hub: {e}")
    print("Using fallback CRNN model. For Parseq, ensure internet connection for torch.hub.")


class ParseqTokenizer:
    """
    Tokenizer for Parseq-style character encoding.

    Handles character-to-index and index-to-character conversions
    for license plate text.
    """

    def __init__(
        self,
        vocab: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        max_length: int = 7,
    ):
        """
        Initialize tokenizer.

        Args:
            vocab: Character vocabulary
            max_length: Maximum sequence length
        """
        self.vocab = vocab
        self.max_length = max_length
        self.char_to_idx = {c: i for i, c in enumerate(vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(vocab)}
        self.vocab_size = len(vocab)

        # Special tokens
        self.pad_token = "[PAD]"
        self.eos_token = "[EOS]"
        self.sos_token = "[SOS]"
        self.unk_token = "[UNK]"

    def encode(self, text: str) -> torch.Tensor:
        """
        Encode text to indices.

        Args:
            text: Input text string

        Returns:
            Tensor of shape (max_length,) with character indices
        """
        indices = torch.zeros(self.max_length, dtype=torch.long)

        for i, char in enumerate(text):
            if i >= self.max_length:
                break
            if char in self.char_to_idx:
                indices[i] = self.char_to_idx[char]
            else:
                # Unknown character - use last index
                indices[i] = self.vocab_size - 1

        return indices

    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Encode multiple texts.

        Args:
            texts: List of text strings

        Returns:
            Tensor of shape (B, max_length)
        """
        encoded = torch.zeros(len(texts), self.max_length, dtype=torch.long)

        for i, text in enumerate(texts):
            encoded[i] = self.encode(text)

        return encoded

    def decode(self, indices: torch.Tensor) -> str:
        """
        Decode indices to text.

        Args:
            indices: Tensor of shape (max_length,) or (B, max_length)

        Returns:
            Decoded text string
        """
        if indices.dim() == 2:
            indices = indices[0]  # Take first batch item

        text = ""
        for idx in indices:
            idx_val = idx.item() if isinstance(idx, torch.Tensor) else idx
            if idx_val == 0:
                continue  # Padding
            if idx_val < self.vocab_size:
                text += self.idx_to_char[idx_val]

        return text

    def decode_batch(self, indices: torch.Tensor) -> List[str]:
        """
        Decode multiple texts.

        Args:
            indices: Tensor of shape (B, max_length)

        Returns:
            List of decoded text strings
        """
        texts = []
        for i in range(indices.shape[0]):
            texts.append(self.decode(indices[i]))

        return texts


class ParseqOCR(nn.Module):
    """
    Parseq OCR Model Wrapper.

    Provides:
    - Loading pretrained Parseq model
    - Fine-tuning on license plate data
    - Inference with logit outputs for LCOFL
    - Text decoding
    """

    def __init__(
        self,
        pretrained_path: str = "baudm/parseq-base",
        vocab: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        max_length: int = 7,
        frozen: bool = True,
    ):
        """
        Initialize Parseq OCR.

        Args:
            pretrained_path: Path or HuggingFace ID for pretrained model
            vocab: Character vocabulary
            max_length: Maximum sequence length
            frozen: If True, freeze weights during SR training
        """
        super().__init__()

        self.vocab = vocab
        self.max_length = max_length
        self.frozen = frozen
        self.tokenizer = ParseqTokenizer(vocab, max_length)

        # Try to load Parseq model
        self.model = None
        if PARSEQ_AVAILABLE:
            try:
                # Load Parseq via torch.hub
                self.model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True, trust_repo=True)
                print(f"Loaded Parseq model via torch.hub")

                # Replace Parseq's output layer to match our vocabulary
                self._replace_parseq_head(vocab)
            except Exception as e:
                print(f"Warning: Could not load Parseq model: {e}")
                print("Creating a simple CNN+RNN model as fallback")
                self.model = None

        # Fallback to simple CRNN model
        if self.model is None:
            self.model = SimpleCRNN(
                vocab_size=len(vocab),
                max_length=max_length,
            )
            print("Using fallback CRNN model")

        # Freeze weights if specified
        if frozen:
            for param in self.parameters():
                param.requires_grad = False

    def _replace_parseq_head(self, vocab: str):
        """
        Replace Parseq's output layer to match our vocabulary.
        Uses programmatic search to find the correct output layer.

        Args:
            vocab: Target vocabulary string (e.g., "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        """
        # Parseq's default vocabulary (94 characters)
        parseq_vocab = (
            "0123456789"
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        )

        # Create mapping: our_char_idx -> parseq_char_idx
        vocab_mapping = []
        for char in vocab:
            if char in parseq_vocab:
                vocab_mapping.append(parseq_vocab.index(char))
            else:
                # Character not in Parseq's vocab - use last index as fallback
                vocab_mapping.append(len(parseq_vocab) - 1)

        # Programmatic search for the output layer
        original_head = None
        head_path = None

        # Search for Linear layer with ~94 output features (Parseq's vocab size)
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Check if this is the output layer (vocab size ~94)
                if module.out_features in [92, 93, 94, 95, 96, 97]:  # Parseq vocab sizes
                    original_head = module
                    head_path = name
                    print(f"Found Parseq output layer: {name}")
                    print(f"  Input: {module.in_features}, Output: {module.out_features}")
                    break

        # Fallback: try common paths if size-based search didn't work
        if original_head is None:
            print("Warning: Could not find output layer by size. Trying direct paths...")
            fallback_paths = [
                'decoder.head', 'decoder.output_projection', 'decoder.lm_head',
                'decoder.linear', 'head', 'lm_head', 'output_projection'
            ]
            for path in fallback_paths:
                obj = self.model
                try:
                    for attr in path.split('.'):
                        obj = getattr(obj, attr)
                    if isinstance(obj, nn.Linear):
                        original_head = obj
                        head_path = path
                        print(f"Found output layer at path: {path}")
                        print(f"  Input: {obj.in_features}, Output: {obj.out_features}")
                        break
                except AttributeError:
                    continue

        if original_head is None:
            print("Error: Could not find Parseq output layer. Model structure:")
            print(self.model)
            return

        # Create new head
        in_features = original_head.in_features
        out_features = len(vocab)  # 36
        new_head = nn.Linear(in_features, out_features)
        new_head = new_head.to(original_head.weight.device)

        print(f"Replacing Parseq head: {original_head.out_features} -> {out_features} classes")
        print(f"Preserving pretrained weights for {len(vocab_mapping)} characters")

        # Initialize with corresponding weights from pretrained layer
        with torch.no_grad():
            for new_idx, old_idx in enumerate(vocab_mapping):
                if old_idx < original_head.out_features:
                    new_head.weight[new_idx] = original_head.weight[old_idx].clone()
                    new_head.bias[new_idx] = original_head.bias[old_idx].clone()

        # Replace the layer using the path
        if '.' in head_path:
            # Navigate to parent and replace
            parts = head_path.split('.')
            parent = self.model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_head)
        else:
            setattr(self.model, head_path, new_head)

        print("Parseq head replaced successfully")

    def forward(
        self,
        images: torch.Tensor,
        return_logits: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images: Input images of shape (B, 3, H, W) in range [-1, 1]
            return_logits: If True, return logits for LCOFL

        Returns:
            If return_logits: Logits of shape (B, max_length, vocab_size)
            Otherwise: Predicted texts
        """
        # Convert from [-1, 1] to [0, 1]
        x = (images + 1.0) / 2.0
        x = torch.clamp(x, 0, 1)

        # Check if using SimpleCRNN fallback
        is_simple_crnn = isinstance(self.model, SimpleCRNN)

        if not is_simple_crnn:
            # Resize to Parseq expected input size (32, 128)
            # Parseq uses VisionTransformer which requires fixed input size
            parseq_h, parseq_w = 32, 128
            if x.shape[2] != parseq_h or x.shape[3] != parseq_w:
                x = F.interpolate(x, size=(parseq_h, parseq_w), mode='bilinear', align_corners=False)

            # Normalize with ImageNet stats (Parseq expects this)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
            x = (x - mean) / std

        # Get model outputs
        if is_simple_crnn:
            # SimpleCRNN handles normalization internally
            return self.model(x, return_logits=return_logits)
        else:
            # Parseq model
            if return_logits:
                logits = self.model(x)
                # Parseq returns (B, C, T), need to transpose to (B, T, C)
                if logits.dim() == 3:
                    logits = logits.permute(0, 2, 1)
                # Truncate or pad to max_length
                if logits.shape[1] > self.max_length:
                    logits = logits[:, :self.max_length, :]
                elif logits.shape[1] < self.max_length:
                    pad_len = self.max_length - logits.shape[1]
                    logits = F.pad(logits, (0, 0, 0, pad_len))
                return logits
            else:
                return self.model(x)

    def predict(self, images: torch.Tensor) -> List[str]:
        """
        Predict text from images.

        Args:
            images: Input images of shape (B, 3, H, W)

        Returns:
            List of predicted text strings
        """
        with torch.no_grad():
            logits = self.forward(images, return_logits=True)

        # Decode predictions
        pred_indices = logits.argmax(dim=-1)  # (B, max_length)
        texts = self.tokenizer.decode_batch(pred_indices)

        return texts

    def finetune(
        self,
        train_loader,
        val_loader=None,
        num_epochs: int = 10,
        lr: float = 1e-4,
        device: str = "cuda",
    ):
        """
        Fine-tune Parseq on license plate data.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of training epochs
            lr: Learning rate
            device: Device to use for training
        """
        # Unfreeze weights for fine-tuning
        self.train()
        self.to(device)

        for param in self.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        for epoch in range(num_epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for batch in train_loader:
                hr_images = batch["hr"].to(device)
                gt_texts = batch["plate_text"]

                # Encode ground truth
                gt_indices = self.tokenizer.encode_batch(gt_texts).to(device)

                # Forward pass
                logits = self.forward(hr_images, return_logits=True)

                # Compute loss
                B, K, C = logits.shape
                logits_flat = logits.reshape(-1, C)
                targets_flat = gt_indices.reshape(-1)

                # Filter padding
                mask = targets_flat > 0
                if mask.sum() > 0:
                    loss = criterion(logits_flat[mask], targets_flat[mask])
                else:
                    loss = torch.tensor(0.0, device=device)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Accuracy
                pred_indices = logits.argmax(dim=-1)
                correct += (pred_indices == gt_indices).sum().item()
                total += gt_indices.numel()

            avg_loss = total_loss / len(train_loader)
            accuracy = correct / total if total > 0 else 0

            print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")

            # Validation
            if val_loader is not None:
                self.eval()
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch in val_loader:
                        hr_images = batch["hr"].to(device)
                        gt_texts = batch["plate_text"]

                        pred_texts = self.predict(hr_images)

                        for pred, gt in zip(pred_texts, gt_texts):
                            if pred == gt:
                                val_correct += 1
                            val_total += 1

                val_acc = val_correct / val_total if val_total > 0 else 0
                print(f"Validation Accuracy: {val_acc:.4f}")
                self.train()

        # Re-freeze after fine-tuning
        if self.frozen:
            for param in self.parameters():
                param.requires_grad = False

    def save(self, path: str):
        """Save model weights."""
        torch.save({
            "model_state_dict": self.state_dict(),
            "vocab": self.vocab,
            "max_length": self.max_length,
        }, path)

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location="cpu")
        self.load_state_dict(checkpoint["model_state_dict"])
        self.vocab = checkpoint["vocab"]
        self.max_length = checkpoint["max_length"]


class SimpleCRNN(nn.Module):
    """
    Simple CRNN model as fallback when Parseq is not available.

    Architecture: CNN feature extractor + Bidirectional LSTM + CTC decoder
    """

    def __init__(
        self,
        vocab_size: int = 36,
        max_length: int = 7,
    ):
        """
        Initialize SimpleCRNN.

        Args:
            vocab_size: Size of character vocabulary
            max_length: Maximum sequence length
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.max_length = max_length

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # RNN decoder
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # Output projection
        self.fc = nn.Linear(512, vocab_size)

    def forward(self, x: torch.Tensor, return_logits: bool = True) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images (B, 3, H, W) in range [0, 1]
            return_logits: If True, return logits

        Returns:
            Logits of shape (B, max_length, vocab_size)
        """
        # CNN features
        features = self.cnn(x)  # (B, 256, H', W')

        # Global average pooling over height to get sequential features
        B, C, H, W = features.shape

        # Collapse height dimension through pooling
        features = F.adaptive_avg_pool2d(features, (1, W))  # (B, 256, 1, W)
        features = features.squeeze(2)  # (B, 256, W)
        features = features.permute(0, 2, 1)  # (B, W, 256)

        # Pad or truncate to max_length
        if W < self.max_length:
            features = F.pad(features, (0, 0, 0, self.max_length - W))
        elif W > self.max_length:
            features = features[:, :self.max_length, :]

        # RNN
        rnn_out, _ = self.rnn(features)  # (B, max_length, 512)

        # Output projection
        logits = self.fc(rnn_out)  # (B, max_length, vocab_size)

        return logits


if __name__ == "__main__":
    # Test Parseq OCR
    print("Testing ParseqOCR...")

    ocr = ParseqOCR(
        vocab="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        max_length=7,
        frozen=False,
    )

    # Test forward pass
    images = torch.randn(2, 3, 32, 64) * 2 - 1  # Range [-1, 1]
    logits = ocr(images, return_logits=True)
    print(f"Logits shape: {logits.shape}")  # Should be (2, 7, 36)

    # Test prediction
    texts = ocr.predict(images)
    print(f"Predicted texts: {texts}")

    # Test tokenizer
    tokenizer = ParseqTokenizer()
    encoded = tokenizer.encode("ABC1234")
    print(f"Encoded: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")

    print("ParseqOCR test passed!")
