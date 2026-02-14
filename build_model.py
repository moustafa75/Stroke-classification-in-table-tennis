# build_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime

class TableTennisDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        # ONLY 3 CLASSES: forehand, backhand, push
        self.class_names = ['forehand', 'backhand', 'push']
        
        # Load images and labels
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(data_dir, class_name)
            if os.path.exists(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_path, img_file))
                        self.labels.append(class_idx)
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {data_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            # Use PIL Image for better compatibility
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms (which should handle PIL Images)
            if self.transform:
                image = self.transform(image)
            else:
                # Default transform if none provided
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                image = transform(image)
        
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = torch.zeros((3, 224, 224))
        
        return image, label

class TableTennisCNN(nn.Module):
    def __init__(self, num_classes=3):  # CHANGED to 3 classes
        super(TableTennisCNN, self).__init__()
        # Enhanced feature extraction with more layers and batch normalization
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.1),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.2),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.3),
            
            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.4),
        )
        
        # Enhanced classifier with better regularization
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

class TableTennisClassifier:
    def __init__(self, dataset_path="table_tennis_dataset"):
        self.dataset_path = dataset_path
        self.splits_path = os.path.join(dataset_path, "splits")
        self.models_path = os.path.join(dataset_path, "models")
        os.makedirs(self.models_path, exist_ok=True)
        
        # ONLY 3 CLASSES
        self.stroke_types = ['forehand', 'backhand', 'push']
        self.num_classes = len(self.stroke_types)
        
        # Check for GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = TableTennisCNN(self.num_classes).to(self.device)
        
        # Enhanced transformations with data augmentation for training
        # IMPORTANT: PIL Image transforms must come BEFORE ToTensor
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Slightly larger for better augmentation
            transforms.RandomCrop(224),  # Random crop for better augmentation
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.33))  # Random erasing for regularization
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def create_data_loaders(self, batch_size=32, num_workers=4):
        """Create data loaders for training and validation"""
        train_path = os.path.join(self.splits_path, "train")
        val_path = os.path.join(self.splits_path, "val")
        
        # Validate paths exist
        if not os.path.exists(train_path):
            raise ValueError(f"Training data path does not exist: {train_path}")
        if not os.path.exists(val_path):
            raise ValueError(f"Validation data path does not exist: {val_path}")
        
        train_dataset = TableTennisDataset(
            train_path, 
            transform=self.train_transform  # Use augmented transform for training
        )
        val_dataset = TableTennisDataset(
            val_path, 
            transform=self.val_transform  # Use basic transform for validation
        )
        
        # Use num_workers for faster data loading (set to 0 on Windows if issues occur)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        return train_loader, val_loader
    
    def train_model(self, epochs=50, batch_size=16):  # Increased epochs
        """Train the model"""
        print("🔄 Creating data loaders...")
        train_loader, val_loader = self.create_data_loaders(batch_size)
        
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        # Enhanced loss function with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Enhanced optimizer with weight decay
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Enhanced learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training history with learning rate tracking
        history = {
            'train_loss': [], 
            'train_acc': [], 
            'val_loss': [], 
            'val_acc': [],
            'learning_rate': [],
            'f1_score': [],
            'precision': [],
            'recall': []
        }
        
        print("🎯 Starting model training...")
        
        best_val_acc = 0.0
        patience = 8  # Early stopping patience
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)
            
            # Validation phase
            val_loss, val_acc, val_f1, val_precision, val_recall = self.validate_model(val_loader, criterion)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            history['train_loss'].append(float(epoch_loss))
            history['train_acc'].append(float(epoch_acc.cpu().numpy()))
            history['val_loss'].append(float(val_loss))
            history['val_acc'].append(float(val_acc.cpu().numpy()))
            history['learning_rate'].append(float(current_lr))
            history['f1_score'].append(float(val_f1))
            history['precision'].append(float(val_precision))
            history['recall'].append(float(val_recall))
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            print(f'  Val F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}')
            print(f'  Learning Rate: {current_lr:.6f}')
            
            scheduler.step()
            
            # Early stopping and save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = os.path.join(self.models_path, 'best_model_3class.pth')
                torch.save(self.model.state_dict(), best_model_path)
                
                # Save additional model info
                model_info = {
                    'epoch': epoch + 1,
                    'val_acc': float(val_acc),
                    'val_f1': float(val_f1),
                    'val_precision': float(val_precision),
                    'val_recall': float(val_recall),
                    'timestamp': datetime.now().isoformat()
                }
                with open(os.path.join(self.models_path, 'best_model_info.json'), 'w') as f:
                    json.dump(model_info, f, indent=2)
                
                patience_counter = 0
                print(f"✅ New best model saved! Validation Accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
                print(f"⏳ No improvement. Patience: {patience_counter}/{patience}")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(self.models_path, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_acc': float(val_acc),
                    'history': history
                }, checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
            
            if patience_counter >= patience:
                print("🛑 Early stopping triggered!")
                break
        
        # Save final model
        torch.save(self.model.state_dict(), os.path.join(self.models_path, 'final_model_3class.pth'))
        
        # Save training history
        history_path = os.path.join(self.models_path, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"💾 Training history saved to {history_path}")
        
        # Plot training history
        self.plot_training_history(history)
        
        return history
    
    def validate_model(self, val_loader, criterion):
        """Validate the model and return metrics"""
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = running_corrects.double() / len(val_loader.dataset)
        
        # Calculate additional metrics
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        val_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        val_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        return val_loss, val_acc, val_f1, val_precision, val_recall
    
    def plot_training_history(self, history):
        """Plot comprehensive training history"""
        fig = plt.figure(figsize=(16, 10))
        
        # Accuracy plot
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(history['train_acc'], label='Training Accuracy', linewidth=2)
        ax1.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(history['train_loss'], label='Training Loss', linewidth=2)
        ax2.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate plot
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(history['learning_rate'], label='Learning Rate', color='green', linewidth=2)
        ax3.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # F1 Score plot
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(history['f1_score'], label='F1 Score', color='orange', linewidth=2)
        ax4.set_title('Validation F1 Score', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('F1 Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Precision plot
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(history['precision'], label='Precision', color='purple', linewidth=2)
        ax5.set_title('Validation Precision', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Precision')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Recall plot
        ax6 = plt.subplot(2, 3, 6)
        ax6.plot(history['recall'], label='Recall', color='red', linewidth=2)
        ax6.set_title('Validation Recall', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Recall')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_path, 'training_history_3class.png'), dpi=300, bbox_inches='tight')
        print(f"📊 Training history plots saved")
        plt.show()
    
    def evaluate_model(self):
        """Evaluate the model on validation set with comprehensive metrics"""
        print("📊 Evaluating model...")
        
        # Load best model
        best_model_path = os.path.join(self.models_path, "best_model_3class.pth")
        if os.path.exists(best_model_path):
            try:
                self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
                print("✅ Loaded best model for evaluation")
                
                # Load model info if available
                info_path = os.path.join(self.models_path, "best_model_info.json")
                if os.path.exists(info_path):
                    with open(info_path, 'r') as f:
                        model_info = json.load(f)
                    print(f"   Model trained for {model_info.get('epoch', 'unknown')} epochs")
                    print(f"   Best validation accuracy: {model_info.get('val_acc', 'unknown'):.4f}")
            except Exception as e:
                print(f"⚠️  Error loading model: {e}")
                print("   Using current model state instead")
        else:
            print("⚠️  Best model not found, using current model state")
        
        _, val_loader = self.create_data_loaders(batch_size=32, num_workers=0)
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate comprehensive metrics
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # Per-class metrics
        f1_per_class = f1_score(all_labels, all_preds, average=None)
        precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
        recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
        
        # Print summary metrics
        print("\n" + "="*60)
        print("📊 EVALUATION METRICS SUMMARY")
        print("="*60)
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Weighted F1 Score: {f1:.4f}")
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall: {recall:.4f}")
        print("\nPer-Class Metrics:")
        for i, stroke in enumerate(self.stroke_types):
            print(f"  {stroke:12s} - F1: {f1_per_class[i]:.4f}, Precision: {precision_per_class[i]:.4f}, Recall: {recall_per_class[i]:.4f}")
        print("="*60)
        
        # Classification report
        print("\n📋 Detailed Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=self.stroke_types, digits=4))
        
        # Confusion matrix
        self.plot_confusion_matrix(all_labels, all_preds)
        
        # Per-class accuracy
        self.plot_class_accuracy(all_labels, all_preds)
        
        return all_preds, all_labels, all_probabilities
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.stroke_types, 
                   yticklabels=self.stroke_types,
                   cbar_kws={'shrink': 0.8})
        plt.title('Confusion Matrix (Forehand, Backhand, Push)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_path, 'confusion_matrix_3class.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_class_accuracy(self, y_true, y_pred):
        """Plot per-class accuracy"""
        cm = confusion_matrix(y_true, y_pred)
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        plt.figure(figsize=(8, 4))
        bars = plt.bar(self.stroke_types, class_accuracy, color=['#2E8B57', '#4169E1', '#FF6347'], alpha=0.7)
        plt.title('Per-Class Accuracy')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, class_accuracy):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_path, 'class_accuracy_3class.png'), dpi=300, bbox_inches='tight')
        plt.show()

def main():
    print("🎾 TABLE TENNIS STROKE CLASSIFIER (3 Classes: Forehand, Backhand, Push)")
    print("=" * 60)
    
    # Initialize classifier
    classifier = TableTennisClassifier()
    
    # Show model architecture
    print("📐 Model Architecture:")
    print(f"Number of classes: {classifier.num_classes}")
    print(f"Classes: {classifier.stroke_types}")
    
    # Train the model
    print("\n🚀 Starting training...")
    history = classifier.train_model(epochs=50, batch_size=16)  # Increased epochs
    
    # Evaluate the model
    print("\n🔍 Evaluating model...")
    classifier.evaluate_model()
    
    print(f"\n✅ Training complete! Models saved in: {classifier.models_path}")

if __name__ == "__main__":
    main()