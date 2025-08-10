# train.py

import os
import argparse
import random
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_util.preprocessor import OCNSpatialSegmentDataset
from model.chemberta_ft_model import MAGChemBERTa
from utils import collate

def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging configuration for training."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': time.time()
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer):
    """Load model checkpoint."""
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']
    return 0, float('inf')


def validate_model(model, dataloader, device, logger):
    """Validate model performance."""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            try:
                token2atom = batch.pop('token2atom')
                batch = {k: v.to(device) for k, v in batch.items()}
                
                mm_pred = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    coords=batch['coords'],
                    token2atom=token2atom,
                    mask=batch['mask'],
                )
                
                mask = batch['mask'].bool()
                loss = F.mse_loss(mm_pred[mask], batch['mm_reduced'][mask])
                
                total_loss += loss.item() * mask.sum().item()
                total_samples += mask.sum().item()
                
            except Exception as e:
                logger.warning(f"Validation batch failed: {e}")
                continue
    
    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss


def train(args):
    # Setup logging and directories
    logger = setup_logging(args.log_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    logger.info("Starting training with spatial segmentation")
    logger.info(f"Arguments: {vars(args)}")
    
    # Set device and seeds
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Prepare dataset filenames
    csv_filenames = []
    for fname in args.filenames:
        if not fname.endswith('.csv'):
            csv_filenames.append(fname + '.csv')
        else:
            csv_filenames.append(fname)
    
    logger.info(f"Loading files: {csv_filenames}")
    
    # Validate input files exist
    for filename in csv_filenames:
        filepath = os.path.join(args.root, filename)
        if not os.path.exists(filepath):
            logger.error(f"Input file not found: {filepath}")
            raise FileNotFoundError(f"Input file not found: {filepath}")
    
    try:
        # Load dataset with spatial segmentation
        ds = OCNSpatialSegmentDataset(
            root=args.root,
            filenames=csv_filenames,
            processed_dir=args.processed_dir,
            augmentations=None,
            max_atoms_per_segment=args.max_atoms_per_segment,
            seed=args.seed,
        )
        logger.info(f"Dataset loaded: {len(ds)} spatial segments")
        
        # Split dataset for training and validation if requested
        if args.val_split > 0:
            val_size = int(len(ds) * args.val_split)
            train_size = len(ds) - val_size
            train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])
            logger.info(f"Dataset split: {train_size} training, {val_size} validation")
        else:
            train_ds, val_ds = ds, None
            logger.info("No validation split used")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    try:
        # Initialize tokenizer and model
        tokenizer = MAGChemBERTa.get_tokenizer(args.pretrained)
        logger.info(f"Tokenizer loaded: {args.pretrained}")

        # Create data loaders
        train_dl = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: collate(x, tokenizer, max_length=args.max_length),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        val_dl = None
        if val_ds is not None:
            val_dl = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=lambda x: collate(x, tokenizer, max_length=args.max_length),
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available()
            )
        
        logger.info(f"Data loaders created: {len(train_dl)} training batches")
        if val_dl:
            logger.info(f"Validation batches: {len(val_dl)}")

        # Initialize model
        model = MAGChemBERTa(
            pretrained_name=args.pretrained, 
            lora_r=args.lora_r
        ).to(device)
        logger.info(f"Model initialized with LoRA r={args.lora_r}")
        
        # Initialize optimizer with weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=args.lr_patience, 
            factor=args.lr_factor,
            verbose=True
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize model/optimizer: {e}")
        raise

    # Load checkpoint if resume requested
    start_epoch = 1
    best_loss = float('inf')
    checkpoint_path = os.path.join(args.checkpoint_dir, 'latest_checkpoint.pt')
    
    if args.resume and os.path.exists(checkpoint_path):
        try:
            start_epoch, best_loss = load_checkpoint(checkpoint_path, model, optimizer)
            start_epoch += 1
            logger.info(f"Resumed from epoch {start_epoch-1}, best loss: {best_loss:.4f}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
    
    # Setup tensorboard logging
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'tensorboard'))
    
    # Training loop
    logger.info("Starting training loop")
    
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_mm = 0
        n_tokens = 0
        epoch_start_time = time.time()
        
        try:
            for batch_idx, batch in enumerate(train_dl):
                try:
                    # Move tensors to device, but keep token2atom as list
                    token2atom = batch.pop('token2atom')
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    mm_pred = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        coords=batch['coords'],
                        token2atom=token2atom,
                        mask=batch['mask'],
                    )

                    mask = batch['mask'].bool()
                    loss = F.mse_loss(mm_pred[mask], batch['mm_reduced'][mask])

                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping for stability
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    
                    optimizer.step()

                    total_mm += loss.item() * mask.sum().item()
                    n_tokens += mask.sum().item()
                    
                    # Log batch metrics
                    if (batch_idx + 1) % args.log_interval == 0:
                        current_loss = total_mm / max(n_tokens, 1)
                        logger.info(
                            f"Epoch {epoch:02d}, Batch {batch_idx+1}/{len(train_dl)}, "
                            f"Loss: {current_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
                        )
                        
                except Exception as e:
                    logger.warning(f"Training batch {batch_idx} failed: {e}")
                    continue
            
            # Calculate epoch metrics
            epoch_loss = total_mm / max(n_tokens, 1)
            epoch_time = time.time() - epoch_start_time
            
            # Validation
            val_loss = None
            if val_dl is not None:
                val_loss = validate_model(model, val_dl, device, logger)
                scheduler.step(val_loss)
                
                # Log to tensorboard
                writer.add_scalar('Loss/Validation', val_loss, epoch)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            else:
                scheduler.step(epoch_loss)
            
            # Log to tensorboard
            writer.add_scalar('Loss/Training', epoch_loss, epoch)
            
            # Print epoch summary
            log_msg = (f"Epoch {epoch:02d} | Train Loss: {epoch_loss:.4f} | "
                      f"Time: {epoch_time:.2f}s")
            if val_loss is not None:
                log_msg += f" | Val Loss: {val_loss:.4f}"
            logger.info(log_msg)
            
            # Save checkpoints
            current_loss = val_loss if val_loss is not None else epoch_loss
            
            # Save latest checkpoint
            save_checkpoint(model, optimizer, epoch, current_loss, checkpoint_path)
            
            # Save best checkpoint
            if current_loss < best_loss:
                best_loss = current_loss
                best_checkpoint_path = os.path.join(args.checkpoint_dir, 'best_checkpoint.pt')
                save_checkpoint(model, optimizer, epoch, current_loss, best_checkpoint_path)
                logger.info(f"New best model saved with loss: {best_loss:.4f}")
            
            # Early stopping
            if hasattr(scheduler, 'num_bad_epochs') and scheduler.num_bad_epochs >= args.early_stop_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            break
        except Exception as e:
            logger.error(f"Epoch {epoch} failed: {e}")
            continue
    
    # Save final model
    final_model_path = os.path.join(args.checkpoint_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")
    
    writer.close()
    logger.info("Training completed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ChemBERTa with OCN spatial segmentation")
    
    # Data arguments
    parser.add_argument('--root', default='./raw', help='Root directory for input CSV files')
    parser.add_argument('--filenames', nargs='+', required=True, 
                        help='List of CSV files with molecule data (extension optional)')
    parser.add_argument('--processed-dir', default='./processed', 
                        help='Directory to save processed data')
    parser.add_argument('--val-split', type=float, default=0.1, 
                        help='Validation split ratio (0.0 to disable validation)')
    
    # Model arguments
    parser.add_argument('--pretrained', default='seyonec/ChemBERTa-zinc-base-v1', 
                        help='Pretrained ChemBERTa model name')
    parser.add_argument('--max-atoms-per-segment', type=int, default=100, 
                        help='Maximum atoms per spatial segment (affects max token length)')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum token sequence length')
    parser.add_argument('--lora-r', type=int, default=8,
                        help='LoRA rank for parameter-efficient fine-tuning')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping threshold (0 to disable)')
    parser.add_argument('--lr-patience', type=int, default=3, help='Patience for learning rate reduction')
    parser.add_argument('--lr-factor', type=float, default=0.5, help='Factor for learning rate reduction')
    parser.add_argument('--early-stop-patience', type=int, default=10, help='Patience for early stopping')
    
    # System arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--log-interval', type=int, default=50, help='Logging interval in batches')
    
    # Checkpoint and logging arguments
    parser.add_argument('--checkpoint-dir', default='./checkpoints', 
                        help='Directory to save model checkpoints')
    parser.add_argument('--log-dir', default='./logs', help='Directory for training logs')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.val_split < 0 or args.val_split >= 1:
        raise ValueError("Validation split must be between 0 and 1")
    
    if args.max_atoms_per_segment <= 0:
        raise ValueError("max_atoms_per_segment must be positive")
    
    try:
        train(args)
    except Exception as e:
        print(f"Training failed: {e}")
        raise