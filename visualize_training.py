# visualize_training.py
# Comprehensive training results visualization script

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
import re

def parse_training_log(log_file_path):
    """Parse training log file to extract metrics from the latest training session only."""
    all_sessions = []
    current_session = {
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'times': [],
        'learning_rates': [],
        'start_line': None
    }
    
    if not os.path.exists(log_file_path):
        print(f"Log file not found: {log_file_path}")
        return None
    
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    # Find all training session starts
    for i, line in enumerate(lines):
        # Detect start of new training session
        if "Starting training with spatial segmentation" in line:
            # Save previous session if it has data
            if current_session['epochs']:
                all_sessions.append(current_session.copy())
            
            # Start new session
            current_session = {
                'epochs': [],
                'train_losses': [],
                'val_losses': [],
                'times': [],
                'learning_rates': [],
                'start_line': i
            }
        
        # Parse epoch summary lines
        elif "Epoch" in line and "Train Loss" in line:
            # Example: "Epoch 01 | Train Loss: 3.7435 | Time: 1.29s | Val Loss: 4.0845"
            epoch_match = re.search(r'Epoch (\d+)', line)
            train_loss_match = re.search(r'Train Loss: ([\d.]+)', line)
            time_match = re.search(r'Time: ([\d.]+)s', line)
            val_loss_match = re.search(r'Val Loss: ([\d.]+)', line)
            
            if epoch_match and train_loss_match:
                current_session['epochs'].append(int(epoch_match.group(1)))
                current_session['train_losses'].append(float(train_loss_match.group(1)))
                
                if time_match:
                    current_session['times'].append(float(time_match.group(1)))
                else:
                    current_session['times'].append(0)
                
                if val_loss_match:
                    current_session['val_losses'].append(float(val_loss_match.group(1)))
                else:
                    current_session['val_losses'].append(None)
        
        # Parse learning rate from batch logs
        elif "LR:" in line and current_session['start_line'] is not None:
            lr_match = re.search(r'LR: ([\d.e-]+)', line)
            if lr_match and len(current_session['learning_rates']) < len(current_session['epochs']):
                current_session['learning_rates'].append(float(lr_match.group(1)))
    
    # Add the last session
    if current_session['epochs']:
        all_sessions.append(current_session)
    
    # Return the latest session
    if all_sessions:
        latest_session = all_sessions[-1]
        print(f"📝 Found {len(all_sessions)} training session(s). Using the latest one with {len(latest_session['epochs'])} epochs.")
        
        # Ensure learning rates match epochs count
        latest_session['learning_rates'] = latest_session['learning_rates'][:len(latest_session['epochs'])]
        
        return latest_session
    else:
        print("❌ No training sessions found in log file.")
        return None

def analyze_checkpoints(checkpoint_dir):
    """Analyze saved checkpoints."""
    checkpoint_info = []
    
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.pt'):
            filepath = os.path.join(checkpoint_dir, filename)
            try:
                checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
                info = {
                    'filename': filename,
                    'epoch': checkpoint.get('epoch', 'Unknown'),
                    'loss': checkpoint.get('loss', 'Unknown'),
                    'timestamp': checkpoint.get('timestamp', None),
                    'file_size_mb': os.path.getsize(filepath) / (1024 * 1024)
                }
                if info['timestamp']:
                    info['datetime'] = datetime.fromtimestamp(info['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                checkpoint_info.append(info)
            except Exception as e:
                print(f"Error loading checkpoint {filename}: {e}")
    
    return sorted(checkpoint_info, key=lambda x: x.get('epoch', 0) if isinstance(x.get('epoch'), int) else 0)

def create_training_visualizations(log_data, checkpoint_info, save_dir='./visualizations'):
    """Create comprehensive training visualizations."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Loss Curves
    ax1 = plt.subplot(2, 3, 1)
    epochs = log_data['epochs']
    train_losses = log_data['train_losses']
    val_losses = log_data['val_losses']
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o')
    if any(v is not None for v in val_losses):
        val_clean = [v for v in val_losses if v is not None]
        val_epochs = [e for e, v in zip(epochs, val_losses) if v is not None]
        plt.plot(val_epochs, val_clean, 'r-', label='Validation Loss', linewidth=2, marker='s')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Training Time per Epoch
    ax2 = plt.subplot(2, 3, 2)
    if log_data['times']:
        plt.bar(epochs, log_data['times'], alpha=0.7, color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.title('Training Time per Epoch')
        plt.grid(True, alpha=0.3)
    
    # 3. Learning Rate Schedule
    ax3 = plt.subplot(2, 3, 3)
    if log_data['learning_rates']:
        plt.plot(epochs[:len(log_data['learning_rates'])], log_data['learning_rates'], 
                'purple', linewidth=2, marker='d')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
    
    # 4. Loss Improvement
    ax4 = plt.subplot(2, 3, 4)
    if len(train_losses) > 1:
        loss_improvements = [train_losses[0] - loss for loss in train_losses]
        plt.plot(epochs, loss_improvements, 'orange', linewidth=2, marker='^')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Improvement from Start')
        plt.title('Cumulative Loss Improvement')
        plt.grid(True, alpha=0.3)
    
    # 5. Checkpoint Information
    ax5 = plt.subplot(2, 3, 5)
    if checkpoint_info:
        checkpoint_names = [info['filename'] for info in checkpoint_info]
        checkpoint_losses = [info['loss'] for info in checkpoint_info if isinstance(info['loss'], (int, float))]
        checkpoint_epochs = [info['epoch'] for info in checkpoint_info if isinstance(info['epoch'], (int, float))]
        
        if checkpoint_losses and checkpoint_epochs:
            plt.scatter(checkpoint_epochs, checkpoint_losses, s=100, alpha=0.7, c='red')
            for i, name in enumerate([info['filename'] for info in checkpoint_info if isinstance(info['loss'], (int, float))]):
                if i < len(checkpoint_epochs):
                    plt.annotate(name.replace('.pt', ''), 
                               (checkpoint_epochs[i], checkpoint_losses[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
            plt.xlabel('Epoch')
            plt.ylabel('Checkpoint Loss')
            plt.title('Saved Checkpoints')
            plt.grid(True, alpha=0.3)
    
    # 6. Training Summary Text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"""Training Summary:
    
Total Epochs: {len(epochs)}
Initial Loss: {train_losses[0]:.4f}
Final Loss: {train_losses[-1]:.4f}
Best Loss: {min(train_losses):.4f}
Loss Reduction: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%

Total Training Time: {sum(log_data['times']):.1f}s
Avg Time/Epoch: {np.mean(log_data['times']):.1f}s

Checkpoints: {len(checkpoint_info)}
"""
    
    if val_losses and any(v is not None for v in val_losses):
        val_clean = [v for v in val_losses if v is not None]
        summary_text += f"Best Val Loss: {min(val_clean):.4f}\n"
    
    plt.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_dir, 'training_visualization.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training visualization saved to: {plot_path}")
    
    # Show the plot
    plt.show()
    
    return plot_path

def print_training_summary(log_data, checkpoint_info):
    """Print a detailed training summary."""
    print("=" * 60)
    print("🚀 MAG-OCN TRAINING RESULTS SUMMARY")
    print("=" * 60)
    
    if log_data:
        epochs = log_data['epochs']
        train_losses = log_data['train_losses']
        val_losses = log_data['val_losses']
        times = log_data['times']
        
        print(f"📊 Training Metrics:")
        print(f"   • Total Epochs: {len(epochs)}")
        print(f"   • Initial Training Loss: {train_losses[0]:.4f}")
        print(f"   • Final Training Loss: {train_losses[-1]:.4f}")
        print(f"   • Best Training Loss: {min(train_losses):.4f}")
        print(f"   • Loss Reduction: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%")
        
        if any(v is not None for v in val_losses):
            val_clean = [v for v in val_losses if v is not None]
            print(f"   • Best Validation Loss: {min(val_clean):.4f}")
            print(f"   • Final Validation Loss: {val_clean[-1]:.4f}")
        
        print(f"\n⏱️ Training Time:")
        print(f"   • Total Time: {sum(times):.1f} seconds ({sum(times)/60:.1f} minutes)")
        print(f"   • Average Time per Epoch: {np.mean(times):.1f} seconds")
        print(f"   • Fastest Epoch: {min(times):.1f} seconds")
        print(f"   • Slowest Epoch: {max(times):.1f} seconds")
    
    if checkpoint_info:
        print(f"\n💾 Saved Checkpoints:")
        for info in checkpoint_info:
            loss_str = f"{info['loss']:.4f}" if isinstance(info['loss'], (int, float)) else str(info['loss'])
            print(f"   • {info['filename']}: Epoch {info['epoch']}, Loss {loss_str}, Size {info['file_size_mb']:.1f}MB")
            if 'datetime' in info:
                print(f"     Saved: {info['datetime']}")
    
    print("\n" + "=" * 60)

def main():
    """Main visualization function."""
    print("🎨 Visualizing MAG-OCN Training Results...")
    
    # Parse training logs
    log_file = './logs/training.log'
    log_data = parse_training_log(log_file)
    
    # Analyze checkpoints
    checkpoint_info = analyze_checkpoints('./checkpoints')
    
    if log_data is None:
        print("❌ Could not parse training logs.")
        return
    
    # Print summary
    print_training_summary(log_data, checkpoint_info)
    
    # Create visualizations
    if log_data['epochs']:
        visualization_path = create_training_visualizations(log_data, checkpoint_info)
        print(f"\n📈 Visualizations created successfully!")
        print(f"🔗 TensorBoard: http://localhost:6006")
        print(f"📁 Plots saved to: {visualization_path}")
    else:
        print("❌ No training data found to visualize.")

if __name__ == "__main__":
    main()
