from pathlib import Path

import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


# 1. Select the exact pretrained weights we want.
# 3.5 million parameters
weights = MobileNet_V2_Weights.IMAGENET1K_V2 #currently I think this is the DEFAULT as well

# 2. Build MobileNetV2 and load those ImageNet weights.
#    Downloads them automatically if they are not already cached.
model = mobilenet_v2(weights=weights)

# 3. Create a folder relative to this script's location.
project_dir = Path(__file__).resolve().parent
save_dir = project_dir / "checkpoints" / "pretrained"
save_dir.mkdir(parents=True, exist_ok=True)

# 4. Choose a filename that identifies the architecture and weights.
save_path = save_dir / "mobilenet_v2_imagenet1k_v2.pth"

# 5. Save the model's current weights and stored statistics.
#    No training has happened, so these are still the ImageNet weights.
torch.save(model.state_dict(), save_path)

# 6. Print confirmation.
parameter_count = sum(p.numel() for p in model.parameters())

print(f"Architecture: MobileNetV2")
print(f"Starting weights: {weights.name}")
print(f"Parameters: {parameter_count:,}")
print(f"Saved to: {save_path}")