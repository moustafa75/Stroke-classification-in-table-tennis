# test_model_persistent_learning.py
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import os
import json
import pickle
from torchvision import transforms
from PIL import Image
from datetime import datetime
import random

# ==============================================
# EXACT SAME MODEL ARCHITECTURE AS TRAINING SCRIPT
# ==============================================
class TableTennisCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(TableTennisCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.4),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 14 * 14, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class PersistentLearner:
    def __init__(self):
        self.stroke_types = ['forehand', 'backhand', 'push']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Setup directories
        self.feedback_dir = "table_tennis_dataset/feedback_corrections"
        self.models_dir = "table_tennis_dataset/models"
        os.makedirs(self.feedback_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        for stroke in self.stroke_types:
            os.makedirs(os.path.join(self.feedback_dir, stroke), exist_ok=True)
        
        # Model files
        self.original_model = os.path.join(self.models_dir, 'best_model_3class.pth')
        self.persistent_model = os.path.join(self.models_dir, 'persistent_model.pth')
        self.memory_file = os.path.join(self.models_dir, 'learning_memory.pkl')
        
        # Initialize model
        self.model = TableTennisCNN(num_classes=3)
        
        # Try to load persistent model first, but validate it
        persistent_loaded = self.load_persistent_model()
        if persistent_loaded:
            # Quick validation test to check if model works
            if self.validate_model():
                print("✅ Loaded persistent model with previous learning")
                self.model_source = "persistent"
            else:
                print("⚠️  Persistent model failed validation, loading original model instead")
                persistent_loaded = False
        
        # Load original model if persistent failed or doesn't exist
        if not persistent_loaded:
            if os.path.exists(self.original_model):
                print("📥 Loading original trained model...")
                try:
                    state_dict = torch.load(self.original_model, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    self.model_source = "original"
                    print("✅ Original model loaded")
                except Exception as e:
                    print(f"❌ Error loading original model: {e}")
                    self.model_source = "random"
            else:
                if self.model_source != "persistent":
                    print("⚠️  No model found, using random weights")
                    self.model_source = "random"
        
        self.model.to(self.device)
        self.model.eval()
        
        # Load learning memory
        self.learning_memory = self.load_memory()
        
        # Transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"\n📊 Learning Memory:")
        print(f"   Total corrections: {self.learning_memory['total_corrections']}")
        for stroke in self.stroke_types:
            count = self.learning_memory['corrections'][stroke]
            print(f"   {stroke}: {count}")
        print(f"   Model source: {self.model_source}")
        
        # Verify model is loaded correctly
        if self.model_source != "random":
            # Test with a dummy input to verify model works
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            try:
                with torch.no_grad():
                    dummy_output = self.model(dummy_input)
                print(f"✅ Model verified: Output shape {dummy_output.shape}, Expected: [1, 3]")
            except Exception as e:
                print(f"❌ Model verification failed: {e}")
    
    def load_persistent_model(self):
        """Try to load the persistent model"""
        if os.path.exists(self.persistent_model):
            try:
                print("🔍 Found persistent model, loading...")
                state_dict = torch.load(self.persistent_model, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                return True
            except Exception as e:
                print(f"❌ Failed to load persistent model: {e}")
        return False
    
    def validate_model(self):
        """Quick validation test to check if model is working correctly"""
        try:
            # Test 1: Check if model can make predictions
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                output = self.model(dummy_input)
                if output.shape != (1, 3):
                    return False
            
            # Test 2: Quick test on a few training samples if available
            train_path = "table_tennis_dataset/splits/train"
            correct = 0
            total = 0
            
            for stroke in self.stroke_types:
                stroke_path = os.path.join(train_path, stroke)
                if not os.path.exists(stroke_path):
                    continue
                    
                images = [os.path.join(stroke_path, f) for f in os.listdir(stroke_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if not images:
                    continue
                
                # Test 1 sample from each class (quick validation)
                test_images = random.sample(images, min(1, len(images)))
                
                for img_path in test_images:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    predicted_stroke, _, _ = self.predict(img)
                    if predicted_stroke == stroke:
                        correct += 1
                    total += 1
                    
                    # Only test 3 samples total for speed
                    if total >= 3:
                        break
                
                if total >= 3:
                    break
            
            # Model is valid if accuracy > 30% on training samples (very lenient to catch completely broken models)
            if total >= 2:
                accuracy = correct / total
                print(f"   Validation test: {correct}/{total} correct ({accuracy*100:.0f}%)")
                return accuracy > 0.3
            else:
                # If no training samples available, just check if model loads
                return True
                
        except Exception as e:
            print(f"   Validation error: {e}")
            return False
    
    def load_memory(self):
        """Load learning memory from disk"""
        default_memory = {
            'total_corrections': 0,
            'corrections': {stroke: 0 for stroke in self.stroke_types},
            'learning_rate': 0.0005,
            'last_correction': None,
            'corrected_images': {}  # Track corrected images: hash -> count
        }
        
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'rb') as f:
                    memory = pickle.load(f)
                # Ensure corrected_images exists for backward compatibility
                if 'corrected_images' not in memory:
                    memory['corrected_images'] = {}
                print(f"📚 Loaded learning memory")
                return memory
            except:
                print("⚠️  Could not load memory, starting fresh")
        
        return default_memory
    
    def _hash_image(self, image):
        """Create a simple hash of the image to track duplicates"""
        import hashlib
        # Convert to grayscale and resize for faster hashing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (64, 64))
        image_bytes = small.tobytes()
        return hashlib.md5(image_bytes).hexdigest()
    
    def save_memory(self):
        """Save learning memory to disk"""
        with open(self.memory_file, 'wb') as f:
            pickle.dump(self.learning_memory, f)
    
    def predict(self, image):
        """Make prediction on image"""
        # Ensure model is in eval mode (important for Dropout and BatchNorm)
        self.model.eval()
        
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        return self.stroke_types[predicted_class], confidence, probabilities.cpu().numpy()
    
    def learn_correction(self, image, correct_stroke, predicted_stroke, current_probs):
        """Learn from a correction with overfitting protection"""
        print(f"\n🧠 LEARNING: '{predicted_stroke}' → '{correct_stroke}'")
        
        # Check if we've corrected this image before
        image_hash = self._hash_image(image)
        correction_count = self.learning_memory['corrected_images'].get(image_hash, 0)
        
        if correction_count >= 2:
            print(f"⚠️  WARNING: This image has been corrected {correction_count} times already!")
            print(f"   Skipping learning to prevent overfitting.")
            print(f"   The model may already be overfitted to this specific image.")
            return current_probs[self.stroke_types.index(correct_stroke)]
        
        correct_idx = self.stroke_types.index(correct_stroke)
        current_correct_prob = current_probs[correct_idx]
        
        # Calculate target probability
        target_prob = min(current_correct_prob + 0.3, 0.9)
        print(f"   Current: {current_correct_prob:.3f} → Target: {target_prob:.3f}")
        
        # Reduce learning iterations and rate for repeated corrections
        if correction_count == 1:
            max_iterations = 5  # Much fewer iterations for second correction
            effective_lr = self.learning_memory['learning_rate'] * 0.3  # Much lower LR
            print(f"   ⚠️  Second correction on this image - using reduced learning (5 iters, LR×0.3)")
        else:
            max_iterations = 15  # Reduced from 20
            effective_lr = self.learning_memory['learning_rate']
        
        # Switch to training mode
        self.model.train()
        
        # Prepare batch for learning
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Create batch with augmentations (IMPORTANT: use actual augmentations)
        batch_size = 4
        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        inputs = []
        for _ in range(batch_size):
            tensor = augment_transform(pil_image).unsqueeze(0)
            inputs.append(tensor)
        
        input_batch = torch.cat(inputs, dim=0).to(self.device)
        targets = torch.tensor([correct_idx] * batch_size, dtype=torch.long).to(self.device)
        
        # Use cross entropy loss
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=effective_lr,
            weight_decay=0.0001  # Add weight decay for regularization
        )
        
        # Track improvements
        best_prob = current_correct_prob
        no_improvement = 0
        new_prob = current_correct_prob  # Initialize to avoid undefined variable
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            outputs = self.model(input_batch)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)  # Reduced from 1.0
            optimizer.step()
            
            # Check progress every few iterations
            if iteration % 2 == 0 or iteration == max_iterations - 1:
                # Temporarily switch to eval mode for proper evaluation
                self.model.eval()
                with torch.no_grad():
                    # Use original transform (no augmentation) for evaluation
                    test_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
                    test_output = self.model(test_tensor)
                    test_probs = torch.softmax(test_output[0], 0)
                    new_prob = test_probs[correct_idx].item()
                    
                    if new_prob > best_prob:
                        best_prob = new_prob
                        no_improvement = 0
                    else:
                        no_improvement += 1
                
                # Switch back to train mode
                self.model.train()
                
                if iteration % 4 == 0 or iteration == 0:
                    print(f"   Iter {iteration}: loss={loss.item():.4f}, prob={new_prob:.3f}")
            
            # Early stopping conditions
            if best_prob >= target_prob:
                print(f"   ✅ Reached target probability {target_prob:.3f}")
                break
            
            if no_improvement >= 4:  # Increased threshold for more stability
                print(f"   ⏹️  No improvement for {no_improvement} checks, stopping")
                break
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            final_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            final_output = self.model(final_tensor)
            final_probs = torch.softmax(final_output[0], 0)
            final_prob = final_probs[correct_idx].item()
            final_pred = torch.argmax(final_output, 1).item()
        
        improvement = final_prob - current_correct_prob
        
        # Update memory
        self.learning_memory['total_corrections'] += 1
        self.learning_memory['corrections'][correct_stroke] += 1
        self.learning_memory['last_correction'] = datetime.now().isoformat()
        self.learning_memory['corrected_images'][image_hash] = correction_count + 1
        
        # Gradually decrease learning rate
        if self.learning_memory['total_corrections'] % 5 == 0:
            self.learning_memory['learning_rate'] *= 0.9
            print(f"   📉 Learning rate decreased to {self.learning_memory['learning_rate']:.6f}")
        
        # Save everything
        self.save_model()
        self.save_memory()
        
        # Save correction image
        self.save_correction_image(image, predicted_stroke, correct_stroke)
        
        print(f"✅ Learning complete!")
        print(f"   Improvement: {current_correct_prob:.3f} → {final_prob:.3f} (+{improvement:.3f})")
        print(f"   Now predicts: '{self.stroke_types[final_pred]}'")
        print(f"   Total corrections: {self.learning_memory['total_corrections']}")
        print(f"   This image corrected {correction_count + 1} time(s)")
        
        return final_prob
    
    def test_on_training_sample(self, num_samples=5):
        """Test the model on actual training samples to verify it's working"""
        train_path = "table_tennis_dataset/splits/train"
        print(f"\n🧪 Testing model on {num_samples} training samples...")
        print("=" * 60)
        
        correct = 0
        total = 0
        
        for stroke in self.stroke_types:
            stroke_path = os.path.join(train_path, stroke)
            if not os.path.exists(stroke_path):
                continue
                
            images = [os.path.join(stroke_path, f) for f in os.listdir(stroke_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not images:
                continue
            
            # Test a few samples from each class
            test_images = random.sample(images, min(num_samples, len(images)))
            
            for img_path in test_images:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                predicted_stroke, confidence, probs = self.predict(img)
                is_correct = predicted_stroke == stroke
                
                if is_correct:
                    correct += 1
                else:
                    print(f"❌ WRONG: {os.path.basename(img_path)}")
                    print(f"   True: {stroke}, Predicted: {predicted_stroke} ({confidence:.3f})")
                
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"\n📊 Results: {correct}/{total} correct ({accuracy*100:.1f}%)")
        
        if accuracy < 0.5:
            print("\n⚠️  WARNING: Model accuracy on training data is very low!")
            print("   This suggests the model may not be loaded correctly.")
            print("   Try option 6 to force reload the original model.")
        
        return accuracy
    
    def save_model(self):
        """Save model persistently"""
        torch.save(self.model.state_dict(), self.persistent_model)
        self.model_source = "persistent"
    
    def save_correction_image(self, image, predicted, correct):
        """Save the corrected image"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{predicted}_to_{correct}.jpg"
        folder = os.path.join(self.feedback_dir, correct)
        filepath = os.path.join(folder, filename)
        cv2.imwrite(filepath, image)
    
    def confirm_prediction(self, image, stroke, confidence):
        """Confirm a correct prediction"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_confirmed_{stroke}_{confidence:.2f}.jpg"
        folder = os.path.join(self.feedback_dir, stroke)
        filepath = os.path.join(folder, filename)
        cv2.imwrite(filepath, image)
        print(f"✅ Confirmed: '{stroke}' ({confidence:.3f})")
    
    def force_reload_original_model(self):
        """Force reload the original trained model, discarding persistent learning"""
        if os.path.exists(self.original_model):
            try:
                print("🔄 Force reloading original model...")
                state_dict = torch.load(self.original_model, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.model_source = "original"
                
                # Clear persistent model
                if os.path.exists(self.persistent_model):
                    print("🗑️  Removing persistent model...")
                    os.remove(self.persistent_model)
                
                print("✅ Original model reloaded successfully!")
                return True
            except Exception as e:
                print(f"❌ Error reloading original model: {e}")
                return False
        else:
            print("❌ Original model not found!")
            return False
    
    def reset_learning(self):
        """Reset to original model"""
        if os.path.exists(self.original_model):
            state_dict = torch.load(self.original_model, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            # Reset memory
            self.learning_memory = {
                'total_corrections': 0,
                'corrections': {stroke: 0 for stroke in self.stroke_types},
                'learning_rate': 0.0005,
                'last_correction': None,
                'corrected_images': {}
            }
            
            # Delete persistent files
            if os.path.exists(self.persistent_model):
                os.remove(self.persistent_model)
            if os.path.exists(self.memory_file):
                os.remove(self.memory_file)
            
            print("🔄 Reset complete! All learning cleared.")
            self.model_source = "original"
            return True
        return False
    
    def test_image(self, image_path):
        """Interactive testing of an image"""
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        image = cv2.imread(image_path)
        if image is None:
            print("❌ Could not load image")
            return
        
        print(f"\n🎯 Testing: {os.path.basename(image_path)}")
        print("=" * 50)
        
        while True:
            stroke, confidence, probs = self.predict(image)
            
            print(f"\n📊 Prediction: '{stroke}' ({confidence:.3f})")
            for i, s in enumerate(self.stroke_types):
                print(f"   {s}: {probs[i]:.3f}")
            
            # Display
            display = image.copy()
            cv2.putText(display, f"Prediction: {stroke}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(display, f"Confidence: {confidence:.3f}", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            y_pos = 130
            for i, s in enumerate(self.stroke_types):
                cv2.putText(display, f"{s}: {probs[i]:.3f}", (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                y_pos += 30
            
            cv2.putText(display, "1:Forehand 2:Backhand 3:Push", (20, y_pos + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display, "SPACE:Confirm | R:Reset | S:Save | Q:Quit", (20, y_pos + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display, f"Total Corrections: {self.learning_memory['total_corrections']}", 
                       (20, y_pos + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            
            cv2.imshow(f'Learning - {os.path.basename(image_path)}', cv2.resize(display, (800, 600)))
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord(' '):  # Confirm
                self.confirm_prediction(image, stroke, confidence)
                print("🔄 Testing again...")
                continue
                
            elif ord('1') <= key <= ord('3'):  # Correct
                correct_idx = key - ord('1')
                correct_stroke = self.stroke_types[correct_idx]
                
                if correct_stroke == stroke:
                    print("⚠️  Same as prediction, confirming instead...")
                    self.confirm_prediction(image, stroke, confidence)
                else:
                    self.learn_correction(image, correct_stroke, stroke, probs)
                    print("🔄 Learning applied!")
                
                continue
                
            elif key == ord('r'):  # Reset
                confirm = input("⚠️  Reset ALL learning? (y/n): ").lower()
                if confirm == 'y':
                    self.reset_learning()
                    print("🔄 Testing again...")
                continue
                
            elif key == ord('s'):  # Save current state
                self.save_model()
                self.save_memory()
                print("💾 Model and memory saved!")
                continue
                
            elif key == ord('q'):  # Quit
                break
        
        cv2.destroyAllWindows()

def main():
    print("🎾 TABLE TENNIS - PERSISTENT LEARNING")
    print("=" * 60)
    print("🧠 Model REMEMBERS learning between sessions")
    print("💾 Persistent file: table_tennis_dataset/models/persistent_model.pth")
    print("📚 Memory file: table_tennis_dataset/models/learning_memory.pkl")
    print("=" * 60)
    
    # Check for original model
    if not os.path.exists("table_tennis_dataset/models/best_model_3class.pth"):
        print("❌ Original model not found!")
        print("   Please run: python build_model_pytorch_updated.py")
        return
    
    # Initialize persistent learner
    try:
        learner = PersistentLearner()
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    while True:
        print("\n" + "=" * 50)
        print("MAIN MENU")
        print("1. Test & Learn (interactive)")
        print("2. Test random validation image")
        print("3. Batch test multiple images")
        print("4. View learning stats")
        print("5. Reset all learning")
        print("6. Force reload original model (discard persistent learning)")
        print("7. Test model on training samples (diagnostic)")
        print("8. Exit")
        
        choice = input("\nChoice (1-8): ").strip()
        
        if choice == "1":
            path = input("Image path: ").strip()
            if os.path.exists(path):
                learner.test_image(path)
            else:
                print("❌ File not found")
                
        elif choice == "2":
            val_path = "table_tennis_dataset/splits/val"
            train_path = "table_tennis_dataset/splits/train"
            
            print("\nSelect source:")
            print("  1. Validation set")
            print("  2. Training set (to test on data model was trained on)")
            source_choice = input("Choice (1 or 2, default 1): ").strip()
            
            if source_choice == "2":
                search_path = train_path
                print("🔍 Searching in TRAINING set (model should get these right!)")
            else:
                search_path = val_path
                print("🔍 Searching in validation set")
            
            images = []
            for stroke in learner.stroke_types:
                stroke_path = os.path.join(search_path, stroke)
                if os.path.exists(stroke_path):
                    imgs = [os.path.join(stroke_path, f) for f in os.listdir(stroke_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    images.extend(imgs)
            
            if images:
                img = random.choice(images)
                true_label = os.path.basename(os.path.dirname(img))
                print(f"🔍 Random image: {os.path.basename(img)}")
                print(f"📁 True label: {true_label}")
                
                # Quick prediction before interactive mode
                test_img = cv2.imread(img)
                if test_img is not None:
                    stroke, confidence, probs = learner.predict(test_img)
                    print(f"🤖 Prediction: {stroke} ({confidence:.3f})")
                    print(f"📊 All probabilities:")
                    for i, s in enumerate(learner.stroke_types):
                        marker = "✅" if s == true_label else "  "
                        print(f"   {marker} {s}: {probs[i]:.3f}")
                    print(f"✅ Correct: {stroke == true_label}\n")
                
                learner.test_image(img)
            else:
                print("❌ No images found")
                
        elif choice == "3":
            folder = input("Folder path (leave empty for validation folder): ").strip()
            if not folder:
                folder = "table_tennis_dataset/splits/val"
            
            if os.path.exists(folder):
                images = []
                for root, dirs, files in os.walk(folder):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            images.append(os.path.join(root, file))
                
                if images:
                    print(f"📁 Found {len(images)} images")
                    random.shuffle(images)
                    
                    for i, img_path in enumerate(images[:10]):  # Test first 10
                        print(f"\n[{i+1}/10] Testing: {os.path.basename(img_path)}")
                        img = cv2.imread(img_path)
                        if img is not None:
                            stroke, confidence, probs = learner.predict(img)
                            print(f"   Prediction: {stroke} ({confidence:.3f})")
                            print(f"   Probs: {dict(zip(learner.stroke_types, [f'{p:.3f}' for p in probs]))}")
                            
                            # Auto-learn if confidence is low
                            if confidence < 0.7:
                                true_label = os.path.basename(os.path.dirname(img_path))
                                if true_label in learner.stroke_types and true_label != stroke:
                                    print(f"   ⚠️  Low confidence, auto-learning {stroke} → {true_label}")
                                    learner.learn_correction(img, true_label, stroke, probs)
                else:
                    print("❌ No images found")
            else:
                print("❌ Folder not found")
                
        elif choice == "4":
            print(f"\n📊 LEARNING STATS:")
            print(f"   Model source: {learner.model_source}")
            print(f"   Total corrections: {learner.learning_memory['total_corrections']}")
            print(f"   Learning rate: {learner.learning_memory['learning_rate']:.6f}")
            print(f"   Last correction: {learner.learning_memory['last_correction'] or 'Never'}")
            print(f"\n   Corrections by stroke:")
            for stroke in learner.stroke_types:
                count = learner.learning_memory['corrections'][stroke]
                print(f"     {stroke}: {count}")
                
        elif choice == "5":
            confirm = input("⚠️  ⚠️  ⚠️  RESET ALL LEARNING? This cannot be undone! (yes/no): ").lower()
            if confirm == 'yes':
                if learner.reset_learning():
                    print("✅ All learning has been reset!")
                else:
                    print("❌ Could not reset")
            else:
                print("Reset cancelled")
                
        elif choice == "6":
            confirm = input("⚠️  Reload original model? This will discard all persistent learning! (y/n): ").lower()
            if confirm == 'y':
                if learner.force_reload_original_model():
                    print("✅ Model reloaded. Testing again...")
                else:
                    print("❌ Failed to reload model")
                    
        elif choice == "7":
            learner.test_on_training_sample(num_samples=5)
            
        elif choice == "8":
            # Save before exit
            learner.save_model()
            learner.save_memory()
            print("💾 Model and memory saved")
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()