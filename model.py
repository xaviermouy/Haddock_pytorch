"""
Modified ResNet18 for single-channel spectrogram classification
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18


def create_resnet18_single_channel(num_classes=2, weights=None):
    """
    Create ResNet18 adapted for single-channel input (spectrograms)

    Args:
        num_classes: Number of output classes (2 for binary classification)
        pretrained: Whether to use pretrained weights (not recommended for single-channel)

    Returns:
        Modified ResNet18 model
    """
    # Create standard ResNet18
    model = resnet18(weights=weights) # ResNet18 architecture (from torchvision): conv1 → bn1 → relu → maxpool → layer1 → layer2 → layer3 → layer4 → avgpool → fc

    # Modify first conv layer to accept 1 channel instead of 3
    # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Modified: Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1 = nn.Conv2d(
        in_channels=1,  # Single channel input
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    # Modify final fully connected layer for binary classification
    # ResNet18 has 512 features before the final FC layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


class SpectrogramClassifier(nn.Module):
    """
    Wrapper class for ResNet18 spectrogram classifier
    """
    def __init__(self, num_classes=2):
        super(SpectrogramClassifier, self).__init__()
        self.model = create_resnet18_single_channel(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # Test the model with a dummy input
    model = SpectrogramClassifier(num_classes=2)
    print(model)

    # Create a dummy spectrogram (batch_size=4, channels=1, height=128, width=128)
    dummy_input = torch.randn(4, 1, 128, 128)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")