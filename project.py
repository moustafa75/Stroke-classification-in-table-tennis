import os
import cv2
import json
import shutil
from tqdm import tqdm
import pandas as pd

class VideoProcessor:
    def __init__(self, base_dir="table_tennis_dataset"):
        self.base_dir = base_dir
        # FIXED: Your stroke folders are INSIDE table_tennis_dataset
        self.raw_videos_dir = base_dir  # Stroke folders are inside table_tennis_dataset
        self.processed_dir = os.path.join(base_dir, "processed")
        self.frames_dir = os.path.join(base_dir, "frames")
        self.models_dir = os.path.join(base_dir, "models")
        
        # Create directory structure
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Stroke types
        self.stroke_types = ['forehand', 'backhand', 'smash', 'push', 'chop', 'serve']

    def extract_frames_simple(self, video_path, output_dir, frames_per_second=3):
        """
        SIMPLIFIED frame extraction
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"    ❌ Cannot open video: {video_path}")
                return 0, []

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Handle cases where fps is 0 or invalid
            if fps <= 0 or fps > 240:
                fps = 30  # Reasonable default
            
            frame_interval = max(1, int(fps / frames_per_second))
            
            # Clean video name for filename
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_name = "".join(c for c in video_name if c.isalnum() or c in ('-', '_')).rstrip()
            
            extracted_frames = []
            saved_count = 0

            print(f"    📊 Video: {total_frames} frames, {fps:.1f} fps")
            
            # Extract frames at intervals
            for i in range(0, total_frames, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    # Resize if too large to save space
                    if frame.shape[1] > 1280:  # if width > 1280
                        frame = cv2.resize(frame, (1280, 720))
                    
                    frame_filename = f"{video_name}_frame_{saved_count:04d}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    
                    # Save frame
                    success = cv2.imwrite(frame_path, frame)
                    if success:
                        extracted_frames.append(frame_filename)
                        saved_count += 1
                else:
                    break

            cap.release()
            return saved_count, extracted_frames

        except Exception as e:
            print(f"    💥 Error processing {video_path}: {e}")
            return 0, []

    def process_all_videos(self, frames_per_second=3):
        """
        Process all videos in the stroke folders
        """
        print("🎬 Starting video processing...")
        dataset_stats = {}
        
        for stroke in self.stroke_types:
            print(f"\n{'='*60}")
            print(f"🎯 Processing {stroke.upper()} videos...")
            print(f"{'='*60}")
            
            # FIXED: Stroke folders are inside table_tennis_dataset
            stroke_video_dir = os.path.join(self.raw_videos_dir, stroke)
            stroke_output_dir = os.path.join(self.frames_dir, stroke)
            
            # Create output directory for this stroke
            os.makedirs(stroke_output_dir, exist_ok=True)
            
            if not os.path.exists(stroke_video_dir):
                print(f"📁 Folder not found: {stroke_video_dir}")
                print(f"💡 Please make sure the '{stroke}' folder exists in {self.base_dir}")
                dataset_stats[stroke] = {'videos_processed': 0, 'total_frames': 0}
                continue
            
            # Find all video files
            video_files = []
            for ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.MP4', '.AVI']:
                video_files.extend([f for f in os.listdir(stroke_video_dir) if f.lower().endswith(ext.lower())])
            
            if not video_files:
                print(f"📭 No videos found in {stroke} folder")
                dataset_stats[stroke] = {'videos_processed': 0, 'total_frames': 0}
                continue
            
            print(f"📹 Found {len(video_files)} videos in {stroke} folder")
            
            stroke_frames = 0
            stroke_videos_processed = 0
            
            for video_file in video_files:
                video_path = os.path.join(stroke_video_dir, video_file)
                print(f"\n  🎥 Processing: {video_file}")
                
                frames_count, frames_list = self.extract_frames_simple(
                    video_path, stroke_output_dir, frames_per_second
                )
                
                if frames_count > 0:
                    stroke_frames += frames_count
                    stroke_videos_processed += 1
                    print(f"    ✅ Extracted {frames_count} frames")
                else:
                    print(f"    ❌ Failed to extract frames")
            
            dataset_stats[stroke] = {
                'videos_processed': stroke_videos_processed,
                'total_frames': stroke_frames
            }
            
            print(f"\n🎉 {stroke.upper()}: {stroke_videos_processed}/{len(video_files)} videos, {stroke_frames} frames")
        
        # Save dataset statistics
        self.save_dataset_stats(dataset_stats)
        return dataset_stats

    def save_dataset_stats(self, stats):
        """Save dataset statistics to JSON and CSV"""
        # JSON file
        stats_file = os.path.join(self.base_dir, "dataset_statistics.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # CSV file for easy viewing
        csv_data = []
        for stroke, data in stats.items():
            csv_data.append({
                'stroke_type': stroke,
                'videos_processed': data['videos_processed'],
                'total_frames': data['total_frames']
            })
        
        df = pd.DataFrame(csv_data)
        csv_file = os.path.join(self.base_dir, "dataset_statistics.csv")
        df.to_csv(csv_file, index=False)
        
        print(f"\n💾 Dataset statistics saved:")
        print(f"   📄 JSON: {stats_file}")
        print(f"   📊 CSV: {csv_file}")

    def check_existing_frames(self):
        """Check how many frames we already have"""
        print("\n🔍 Checking existing frames...")
        existing_stats = {}
        
        for stroke in self.stroke_types:
            stroke_dir = os.path.join(self.frames_dir, stroke)
            if os.path.exists(stroke_dir):
                frames = [f for f in os.listdir(stroke_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                existing_stats[stroke] = len(frames)
                print(f"   {stroke:12}: {len(frames):4} frames")
            else:
                existing_stats[stroke] = 0
                print(f"   {stroke:12}: 0 frames")
        
        return existing_stats

    def create_train_val_split(self, train_ratio=0.8):
        """
        Create training and validation splits
        """
        print("\n🔄 Creating train/validation splits...")
        
        splits_dir = os.path.join(self.base_dir, "splits")
        os.makedirs(os.path.join(splits_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(splits_dir, "val"), exist_ok=True)
        
        split_info = {}
        
        for stroke in self.stroke_types:
            stroke_dir = os.path.join(self.frames_dir, stroke)
            if not os.path.exists(stroke_dir):
                continue
            
            frames = [f for f in os.listdir(stroke_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not frames:
                continue
            
            # Shuffle frames
            import random
            random.shuffle(frames)
            
            # Split
            split_idx = int(len(frames) * train_ratio)
            train_frames = frames[:split_idx]
            val_frames = frames[split_idx:]
            
            # Create stroke folders in splits
            train_stroke_dir = os.path.join(splits_dir, "train", stroke)
            val_stroke_dir = os.path.join(splits_dir, "val", stroke)
            os.makedirs(train_stroke_dir, exist_ok=True)
            os.makedirs(val_stroke_dir, exist_ok=True)
            
            # Copy files to splits
            for frame in train_frames:
                src = os.path.join(stroke_dir, frame)
                dst = os.path.join(train_stroke_dir, frame)
                shutil.copy2(src, dst)
            
            for frame in val_frames:
                src = os.path.join(stroke_dir, frame)
                dst = os.path.join(val_stroke_dir, frame)
                shutil.copy2(src, dst)
            
            split_info[stroke] = {
                'train': len(train_frames),
                'val': len(val_frames),
                'total': len(frames)
            }
            
            print(f"   {stroke:12}: {len(train_frames):3} train, {len(val_frames):3} val")
        
        # Save split information
        split_file = os.path.join(self.base_dir, "data_splits.json")
        with open(split_file, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print("✅ Train/validation splits created!")
        return split_info

# MAIN EXECUTION
def main():
    print("🎬 TABLE TENNIS VIDEO PROCESSING - CORRECTED VERSION")
    print("=" * 65)
    print("Stroke classes: forehand, backhand, smash, push, chop, serve")
    print("=" * 65)
    
    # Show current directory and contents
    print(f"📁 Current directory: {os.getcwd()}")
    
    # Check what's inside table_tennis_dataset
    dataset_path = "table_tennis_dataset"
    print(f"\n📂 Contents of {dataset_path}/:")
    if os.path.exists(dataset_path):
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}/")
                # Show contents of stroke folders
                if item in ['forehand', 'backhand', 'smash', 'push', 'chop', 'serve']:
                    try:
                        videos = [f for f in os.listdir(item_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                        print(f"     🎥 {len(videos)} videos")
                        for video in videos[:2]:  # Show first 2 videos
                            print(f"       - {video}")
                        if len(videos) > 2:
                            print(f"       ... and {len(videos) - 2} more")
                    except:
                        print(f"     (could not read contents)")
            else:
                print(f"  📄 {item}")
    else:
        print("❌ table_tennis_dataset folder not found!")
        return
    
    # Initialize processor
    processor = VideoProcessor()
    
    # Check existing frames first
    print("\nStep 1: Checking existing data...")
    existing = processor.check_existing_frames()
    
    # Process videos
    print("\nStep 2: Extracting frames from videos...")
    stats = processor.process_all_videos(frames_per_second=4)
    
    # Create data splits
    print("\nStep 3: Creating training/validation splits...")
    splits = processor.create_train_val_split()
    
    # Print final summary
    print("\n" + "=" * 65)
    print("🎉 PROCESSING COMPLETE!")
    print("=" * 65)
    
    total_frames = 0
    print("\n📊 Final breakdown:")
    for stroke in processor.stroke_types:
        if stroke in stats and 'total_frames' in stats[stroke]:
            frames_count = stats[stroke]['total_frames']
            total_frames += frames_count
            print(f"   {stroke:12}: {frames_count:4} frames")
        else:
            print(f"   {stroke:12}:    0 frames")
    
    print(f"\n📦 Total frames in dataset: {total_frames}")
    print(f"📁 Dataset location: {processor.base_dir}")
    
    # Final check
    if total_frames == 0:
        print("\n⚠️  WARNING: No frames were extracted!")
        print("💡 Check that your stroke folders contain video files")
        print("   Expected structure: table_tennis_dataset/forehand/your_video.mp4")

if __name__ == "__main__":
    main()