import os
import platform
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def get_script_directory():
    """Restituisce il percorso assoluto della cartella dove si trova questo script."""
    return Path(__file__).resolve().parent

def get_best_font(size):
    """
    Cerca un font carino compatibile con il sistema operativo in uso.
    Supporta Windows, macOS e Linux.
    """
    system = platform.system()
    
    # Lista di font candidati per ogni OS
    candidates = []
    
    if system == "Windows":
        candidates = ["arial.ttf", "calibri.ttf", "seguiemj.ttf"]
    elif system == "Darwin": # macOS
        candidates = [
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Verdana.ttf"
        ]
    else: # Linux
        candidates = [
            "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "FreeSans.ttf"
        ]
    
    # Aggiungiamo anche dei fallback generici
    candidates.extend(["arial.ttf", "DejaVuSans.ttf"])

    for font_name in candidates:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
            
    print("⚠️ Nessun font di sistema trovato. Uso il font di default (sarà piccolo/pixellato).")
    return ImageFont.load_default()

def frame_and_label_collage(input_filename, output_filename):
    # Costruiamo i percorsi completi basati sulla posizione dello script
    base_dir = get_script_directory()
    image_path = base_dir / input_filename
    output_path = base_dir / output_filename

    print(f"📂 Cartella di lavoro: {base_dir}")
    print(f"🔍 Cerco immagine: {image_path}")

    if not image_path.exists():
        print(f"❌ ERRORE: Il file '{input_filename}' non esiste in questa cartella!")
        # Creiamo un file dummy solo per test se manca l'immagine
        print("   -> Creo un'immagine di test per farti vedere il risultato...")
        Image.new('RGB', (900, 300), color='gray').save(image_path)

    try:
        # 1. Carica l'immagine
        img = Image.open(image_path)
        orig_w, orig_h = img.size
        
        # 2. Configurazione Stile
        border_size = 30           
        header_height = 100        
        bg_color = "white"
        text_color = "black"
        
        # 3. Nuova Immagine
        new_w = orig_w + (border_size * 2)
        new_h = orig_h + header_height + border_size
        new_img = Image.new("RGB", (new_w, new_h), bg_color)
        
        # 4. Incolla l'originale
        new_img.paste(img, (border_size, header_height))
        
        # 5. Carica il Font (Cross-Platform)
        # Dimensione proporzionale (1/30 della larghezza)
        font_size = max(20, int(orig_w / 30))
        font = get_best_font(font_size)
        
        # 6. Scrittura Etichette
        draw = ImageDraw.Draw(new_img)
        labels = ["Input", "Risultato", "Target"]
        section_width = orig_w / 3
        
        for i, label in enumerate(labels):
            # Calcolo centro sezione
            center_x = (i * section_width) + (section_width / 2) + border_size
            
            # Calcolo dimensione testo
            length = draw.textlength(label, font=font)
            
            # Coordinate
            text_x = center_x - (length / 2)
            text_y = (header_height / 2) - (font_size * 0.7) # *0.7 per centrare meglio visivamente
            
            draw.text((text_x, text_y), label, fill=text_color, font=font)

        # 7. Salvataggio
        new_img.save(output_path)
        print(f"✅ Fatto! Immagine salvata in:\n   {output_path}")

    except Exception as e:
        print(f"❌ Errore imprevisto: {e}")

# ================= CONFIGURAZIONE =================
if __name__ == "__main__":
    # Sostituisci SOLO questo nome con il tuo file.
    # Non serve mettere percorsi tipo C:\..., basta il nome del file
    # se questo script è nella stessa cartella.
    FILE_INPUT = "e.png"   
    FILE_OUTPUT = "image_labeled_final.jpg"
    
    frame_and_label_collage(FILE_INPUT, FILE_OUTPUT)