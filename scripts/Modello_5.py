import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

CURRENT_SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
ROOT_DATA_DIR = PROJECT_ROOT / "data"

TARGET_NAME = "M42"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / TARGET_NAME / "test_results_tiff"
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / TARGET_NAME / "final_weights" / "best.pth"

try:
    from src.architecture import HybridSuperResolutionModel
    from src.dataset import AstronomicalDataset
    from src.metrics import Metrics
except ImportError:
    sys.exit("❌ Errore Import src.")

def save_as_tiff16(tensor, path):
    arr = tensor.squeeze().float().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    arr_u16 = (arr * 65535).astype(np.uint16)
    Image.fromarray(arr_u16, mode='I;16').save(path)

def run_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🧪 INFERENZA: {TARGET_NAME}")
    
    if not CHECKPOINT_PATH.exists():
        print(f"❌ Modello non trovato: {CHECKPOINT_PATH}")
        print("   Esegui Modello_4.py")
        return

    (OUTPUT_DIR / "tiff_science").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "png_preview").mkdir(parents=True, exist_ok=True)

    splits_dir = ROOT_DATA_DIR / TARGET_NAME / "8_dataset_split" / "splits_json"
    test_json = splits_dir / "test.json"
    
    if not test_json.exists():
        test_json = splits_dir / "val.json"
        print("⚠️ Uso Validation set")

    test_ds = AstronomicalDataset(test_json, base_path=PROJECT_ROOT, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
    
    print("   Caricamento Modello...")
    model = HybridSuperResolutionModel(smoothing='balanced', device=device, output_size=512).to(device)
    
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print("   ✅ Pesi caricati")
    except Exception as e:
        print(f"   ❌ Errore: {e}")
        return

    model.eval()
    metrics = Metrics()
    
    print("   🚀 Generazione...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            
            with torch.cuda.amp.autocast():
                sr = model(lr)
            
            metrics.update(sr.float(), hr.float())
            
            save_as_tiff16(sr, OUTPUT_DIR / "tiff_science" / f"sr_{i:04d}.tiff")
            
            lr_up = torch.nn.functional.interpolate(lr, size=(512,512), mode='nearest')
            comp = torch.cat((lr_up, sr, hr), dim=3).clamp(0,1)
            save_image(comp, OUTPUT_DIR / "png_preview" / f"comp_{i:04d}.png")

    res = metrics.compute()
    print("\n📊 RISULTATI:")
    print(f"   PSNR: {res['psnr']:.2f} dB")
    print(f"   SSIM: {res['ssim']:.4f}")
    print(f"📂 Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_test()