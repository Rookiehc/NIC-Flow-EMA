import re
import argparse
import os
import csv
import sys

def parse_log(log_path, output_dir):
    train_data = []
    val_data = []

    # Regex patterns
    # Example: [IMM][it 1/200000] loss=0.5900 ... PSNR=16.36dB, SSIM=0.8136 ...
    # Updated to capture sub-losses: loss_mmd=..., ldif=..., adv_g=..., llpips=..., lhr=...
    train_pattern = re.compile(r'\[IMM\]\[it (\d+)/\d+\] loss=([\d\.]+).*?loss_mmd=([\d\.]+), ldif=([\d\.]+)\*[\d\.]+, (?:ldis|adv_g)=([\d\.]+)\*[\d\.]+, llpips=([\d\.]+)\*[\d\.]+, lhr=([\d\.]+)\*[\d\.]+.*?PSNR=([\d\.]+)dB, SSIM=([\d\.]+)')
    
    # Example: [IMM][val step 5000] L1=0.0270, PSNR=26.55dB, SSIM=0.9837, LPIPS=0.0073 ...
    val_pattern = re.compile(r'\[IMM\]\[val step (\d+)\] .*?PSNR=([\d\.]+)dB, SSIM=([\d\.]+)')

    print(f"Parsing log file: {log_path}")
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                # Check for training line
                train_match = train_pattern.search(line)
                if train_match:
                    iteration = int(train_match.group(1))
                    loss_total = float(train_match.group(2))
                    loss_mmd = float(train_match.group(3))
                    ldif = float(train_match.group(4))
                    adv_g = float(train_match.group(5))
                    llpips = float(train_match.group(6))
                    lhr = float(train_match.group(7))
                    psnr = float(train_match.group(8))
                    ssim = float(train_match.group(9))
                    train_data.append({
                        'iteration': iteration, 
                        'loss': loss_total, 
                        'loss_mmd': loss_mmd,
                        'ldif': ldif,
                        'adv_g': adv_g,
                        'llpips': llpips,
                        'lhr': lhr,
                        'psnr': psnr, 
                        'ssim': ssim
                    })
                    continue

                # Check for validation line
                val_match = val_pattern.search(line)
                if val_match:
                    step = int(val_match.group(1))
                    psnr = float(val_match.group(2))
                    ssim = float(val_match.group(3))
                    val_data.append({'iteration': step, 'val_psnr': psnr, 'val_ssim': ssim})
    except Exception as e:
        print(f"Error reading log file: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV using standard library csv module
    if train_data:
        train_csv_path = os.path.join(output_dir, 'train_metrics.csv')
        with open(train_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['iteration', 'loss', 'loss_mmd', 'ldif', 'adv_g', 'llpips', 'lhr', 'psnr', 'ssim'])
            writer.writeheader()
            writer.writerows(train_data)
        print(f"Saved training metrics to {train_csv_path} ({len(train_data)} records)")
    else:
        print("No training metrics found in log.")
    
    if val_data:
        val_csv_path = os.path.join(output_dir, 'val_metrics.csv')
        with open(val_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['iteration', 'val_psnr', 'val_ssim'])
            writer.writeheader()
            writer.writerows(val_data)
        print(f"Saved validation metrics to {val_csv_path} ({len(val_data)} records)")
    else:
        print("No validation metrics found in log.")

    # Try plotting if libraries are available
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        
        if not train_data:
            return

        df_train = pd.DataFrame(train_data)
        df_val = pd.DataFrame(val_data) if val_data else pd.DataFrame(columns=['iteration', 'val_psnr', 'val_ssim'])

        plt.figure(figsize=(12, 10))
        
        # Loss
        plt.subplot(3, 1, 1)
        # Plot sub-losses
        plt.plot(df_train['iteration'], df_train['ldif'], label='Train ldif', alpha=0.7, linewidth=1)
        plt.plot(df_train['iteration'], df_train['adv_g'], label='Train adv_g', alpha=0.7, linewidth=1)
        plt.plot(df_train['iteration'], df_train['lhr'], label='Train lhr', alpha=0.7, linewidth=1)
        plt.plot(df_train['iteration'], df_train['llpips'], label='Train llpips', alpha=0.7, linewidth=1)
        # Optional: Plot total loss if needed, or keep separate
        # plt.plot(df_train['iteration'], df_train['loss'], label='Total Loss', color='black', alpha=0.3, linestyle='--')
        
        plt.title('Training Losses (Decomposed)')
        plt.xlabel('Iteration')
        plt.ylabel('Loss Value')
        plt.yscale('log') # Log scale is often useful for losses
        plt.legend()
        plt.grid(True, alpha=0.3, which='both')

        # PSNR
        plt.subplot(3, 1, 2)
        plt.plot(df_train['iteration'], df_train['psnr'], label='Train PSNR', color='green', alpha=0.5, linewidth=1)
        if not df_val.empty:
            plt.plot(df_val['iteration'], df_val['val_psnr'], label='Val PSNR', color='red', marker='o', linestyle='-')
        plt.title('PSNR (Peak Signal-to-Noise Ratio)')
        plt.xlabel('Iteration')
        plt.ylabel('dB')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # SSIM
        plt.subplot(3, 1, 3)
        plt.plot(df_train['iteration'], df_train['ssim'], label='Train SSIM', color='purple', alpha=0.5, linewidth=1)
        if not df_val.empty:
            plt.plot(df_val['iteration'], df_val['val_ssim'], label='Val SSIM', color='orange', marker='o', linestyle='-')
        plt.title('SSIM (Structural Similarity Index)')
        plt.xlabel('Iteration')
        plt.ylabel('SSIM')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'metrics_plot.png')
        plt.savefig(plot_path)
        print(f"Saved plot to {plot_path}")
        
    except ImportError:
        print("matplotlib or pandas not found. Skipping plot generation. CSV files were saved.")
    except Exception as e:
        print(f"Error generating plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse training log and extract metrics.")
    parser.add_argument("log_path", type=str, help="Path to the train.log file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save output files (defaults to log file directory)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_path):
        print(f"Error: Log file not found at {args.log_path}")
        sys.exit(1)
        
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.dirname(args.log_path)
        
    parse_log(args.log_path, output_dir)
