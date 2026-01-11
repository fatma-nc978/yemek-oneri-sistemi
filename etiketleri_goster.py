import pandas as pd
import ast
from collections import Counter
import os

file_path = "reindexed/recipes_small_reindexed_translated.csv"

print(f" '{file_path}' dosyası okunuyor...")

if not os.path.exists(file_path):
    print(" HATA: Dosya bulunamadı! Lütfen 'reindexed' klasörünün doğru yerde olduğundan emin ol.")
else:
    # Sadece etiket sütununu okuyalım, hızlı olsun
    df = pd.read_csv(file_path, usecols=["tags_tr"])

    all_tags = []
    
    print(" Etiketler analiz ediliyor...")

    for raw_tags in df["tags_tr"].dropna():
        try:
            # CSV'de etiketler "['etiket1', 'etiket2']" şeklinde string olarak durur.
            # Bunları gerçek listeye çeviriyoruz.
            if isinstance(raw_tags, str):
                tags_list = ast.literal_eval(raw_tags)
                
                # Her etiketi küçük harfe çevirip temizleme
                for t in tags_list:
                    all_tags.append(t.strip().lower())
                    
        except:
            continue

    # En çok geçen 50 etiketi sayma
    tag_counts = Counter(all_tags).most_common(50)

    print("\n" + "="*40)
    print(" EN ÇOK KULLANILAN 50 TÜRKÇE ETİKET")
    print("="*40)
    
    for tag, count in tag_counts:
        print(f" {tag} ({count} tarifte var)")

    print("\n" + "="*40)
    print(" İPUCU: Yukarıdaki listeye bakarak app.py içindeki TAG_MAP kısmını güncelle!")