import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import sys
import os
import matplotlib.pyplot as plt
from model import SimpleFNN

def predict_image(image_path, model_path):
    # 1. Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleFNN().to(device)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please run train.py first.")
        return

    print(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Image Preprocessing
    # Load image
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
        
    print(f"Processing image: {image_path}")
    img_orig = Image.open(image_path).convert('L') # Convert to grayscale
    
    # Auto-invert strategy:
    # Check if the corners are white (high value). If so, it's likely black digit on white paper.
    # MNIST is white digit (high value) on black background (low value).
    # We want input to be like MNIST.
    # If image is white background, we invert.
    # Simple check: average pixel intensity. If > 127, likely white background.
    
    stat = ImageOps.grayscale(img_orig).getextrema()
    # A more robust way: check average of corners? Or just assume standard "black ink on white paper" photo
    # User instruction says: "possible need to reverse grayscale".
    # We'll assume white paper (255) and black ink (0), so we need to invert to get Black bg (0) and White ink (255).
    
    # Let's just allow the user to see both or force invert.
    # Usually photos are white background.
    img_inverted = ImageOps.invert(img_orig)
    
    # Resize to 28x28
    # Using BILINEAR or LANCZOS for better quality downsampling
    img_resized = img_inverted.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Transform to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    img_tensor = transform(img_resized).unsqueeze(0).to(device) # Add batch dim
    
    # 3. Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)
        
    # 4. Visualize
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_orig, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(img_inverted, cmap='gray')
    plt.title("Inverted (Input to Resize)")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img_resized, cmap='gray')
    plt.title(f"Model Input (28x28)\nPred: {predicted.item()} ({conf.item():.2%})")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Prediction: {predicted.item()} with confidence {conf.item():.4f}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_path = os.path.join(script_dir, 'models', 'mnist_fnn.pth')
    
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        predict_image(img_path, default_model_path)
    else:
        print("Usage: python predict_photo.py <path_to_image>")
        print("Please provide an image path.")

