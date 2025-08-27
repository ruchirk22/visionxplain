# src/explainability_module.py

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import json
import matplotlib.pyplot as plt

from data_ingestion import IMAGE_TRANSFORMS, DATA_DIR
from cv_module import CVModule, CV_LOG_PATH
from nlp_module import NLPModule  # still loaded for consistency/logging

from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

# --- Configuration ---
EXPLANATIONS_DIR = DATA_DIR / "explanations"
EXPLANATIONS_DIR.mkdir(exist_ok=True)


# ---- Simple CV wrapper that returns class logits (differentiable) ----
class CVLogitModel(torch.nn.Module):
    def __init__(self, cv_model):
        super().__init__()
        self.model = cv_model

    def forward(self, image_tensor):
        # Expect shape: [B, C, H, W]; returns [B, num_classes] logits
        return self.model(image_tensor)


if __name__ == "__main__":
    print("--- Feature 4: Explainability Module (Corrected, CV-Attribution) ---")

    # 1) Device & modules
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    cv_module = CVModule(device=device)
    nlp_module = NLPModule(device=device)  # optional; not used in attribution graph

    # Ensure eval() to disable dropout etc.
    cv_module.model.eval()

    # 2) Pick a sample image from logs
    try:
        with open(DATA_DIR / "traceability_log.json", "r") as f:
            trace_log = {item["image_id"]: item for item in json.load(f)}
        with open(CV_LOG_PATH, "r") as f:
            cv_log = json.load(f)
        if not trace_log or not cv_log:
            raise FileNotFoundError

        valid_entry = None
        for cv_entry in cv_log:
            if cv_entry["image_id"] in trace_log:
                valid_entry = cv_entry
                break

        if not valid_entry:
            print("No matching image IDs found between traceability and CV logs. Exiting.")
            raise SystemExit(0)

        sample_image_id = valid_entry["image_id"]
        image_path = Path(trace_log[sample_image_id]["local_path"])

    except (FileNotFoundError, json.JSONDecodeError):
        print("A log file is missing or empty. Please run data_ingestion.py and cv_module.py first. Exiting.")
        raise SystemExit(0)

    print(f"\nExplaining first valid image from logs: {image_path} (ID: {sample_image_id})")

    # 3) Preprocess image (use 224x224; requires_grad for attribution)
    original_img = Image.open(image_path).convert("RGB")
    original_img = original_img.resize((224, 224))
    img_tensor = IMAGE_TRANSFORMS(original_img).unsqueeze(0).to(device)
    img_tensor.requires_grad_(True)

    # 4) Choose target class from CV log (word → class index)
    #    Fall back to top-1 prediction if mapping is missing.
    target_word = valid_entry["detections"][0]["object"]
    reverse_class_map = {name: idx for idx, name in cv_module.idx_to_class.items()}
    if target_word in reverse_class_map:
        target_class_idx = reverse_class_map[target_word]
        print(f"Explaining the CV class for word '{target_word}' (class idx: {target_class_idx})")
    else:
        print(f"Warning: '{target_word}' not in CV classes. Falling back to top-1 prediction.")
        with torch.no_grad():
            tmp_logits = cv_module.model(img_tensor)
            target_class_idx = int(torch.argmax(tmp_logits, dim=1).item())
            target_word = cv_module.idx_to_class[target_class_idx]
        print(f"Using top-1 CV class: '{target_word}' (class idx: {target_class_idx})")

    # 5) Integrated Gradients on the CV model
    model_for_attr = CVLogitModel(cv_module.model)
    ig = IntegratedGradients(model_for_attr)

    print("Generating attribution map with Integrated Gradients (CV target)...")
    attributions, delta = ig.attribute(
        inputs=img_tensor,
        baselines=torch.zeros_like(img_tensor),
        target=target_class_idx,
        return_convergence_delta=True,
        n_steps=150,
    )
    print(f"Convergence Delta: {delta.item():.4f} (closer to 0 is better)")

    # 6) Visualize & save
    # attributions: [1, 3, 224, 224] → HxWx3
    attribution_map = attributions.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    original_img_np = np.array(original_img)

    print("Visualizing and saving the saliency map...")
    fig, _ = viz.visualize_image_attr(
        attribution_map,
        original_img_np,
        method="blended_heat_map",
        sign="absolute_value",
        show_colorbar=True,
        title=f"CV Explanation for class: '{target_word}'",
    )

    output_path = EXPLANATIONS_DIR / f"explanation_{sample_image_id}_{target_word}.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nExplanation saved to: {output_path}")

    # 7) Verification
    print("\nVerifying the output...")
    if output_path.exists():
        print("Successfully created and saved explanation image.")
        print("PRD Check: Saliency map visualization has been generated.")
    else:
        print("Failed to save the explanation image.")

    print("\n--- Explainability Module Feature Complete (CV-Attribution) ---")
