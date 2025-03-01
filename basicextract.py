import json
from openai import OpenAI
import base64
import os
import tempfile
import traceback
import time
from pypdf import PdfReader
from olmocr.pipeline import build_page_query, PageResult

# PageResponse sınıfı
class CustomPageResponse:
    """
    Basitleştirilmiş PageResponse sınıfı
    """
    def __init__(self, primary_language="tr", natural_text="", is_rotation_valid=True, 
                 rotation_correction=0, is_table=False, is_diagram=False):
        self.primary_language = primary_language
        self.natural_text = natural_text
        self.is_rotation_valid = is_rotation_valid
        self.rotation_correction = rotation_correction
        self.is_table = is_table
        self.is_diagram = is_diagram

async def process_pdf_as_image(client, filename, page_num, max_tokens=1000):
    """
    PDF sayfasını görüntüye dönüştürüp doğrudan işle
    """
    # Geçici bir dosya oluştur
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        temp_img_path = temp_file.name
    
    try:
        # PDF'i görüntüye dönüştür - önce PyMuPDF dene, sonra diğerlerini
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(filename)
            if page_num <= len(doc):
                page = doc[page_num-1]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                pix.save(temp_img_path)
                print(f"PyMuPDF ile görüntü oluşturuldu")
            else:
                raise ValueError(f"PDF dosyası sadece {len(doc)} sayfa içeriyor")
        except ImportError:
            try:
                from pdf2image import convert_from_path
                images = convert_from_path(filename, first_page=page_num, last_page=page_num)
                if images:
                    images[0].save(temp_img_path, 'PNG')
                    print(f"pdf2image ile görüntü oluşturuldu")
                else:
                    raise Exception("Görüntü oluşturulamadı")
            except ImportError:
                import subprocess
                cmd = f"magick convert -density 300 -quality 100 {filename}[{page_num-1}] {temp_img_path}"
                subprocess.run(cmd, shell=True, check=True)
                print(f"ImageMagick ile görüntü oluşturuldu")
        
        # Görüntüyü base64'e dönüştür
        with open(temp_img_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Görüntüyü işle
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                        "detail": "high"
                    }
                },
                {
                    "type": "text",
                    "text": "Bu görseldeki metni tanı ve JSON formatında döndür. Eğer bir tahsilat makbuzu ise, makbuz numarası, tarih, tutar ve diğer önemli bilgileri çıkar."
                }
            ]
        }]
        
        response = client.chat.completions.create(
            model="allenai_olmocr-7b-0225-preview",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1
        )
        
        # Geçici dosyayı temizle
        try:
            os.remove(temp_img_path)
        except:
            pass
        
        return response
    except Exception as e:
        print(f"Görüntü işleme hatası: {e}")
        # Geçici dosyayı temizlemeyi dene
        try:
            os.remove(temp_img_path)
        except:
            pass
        raise

async def process_page_section(filename, page_num, section_name="tam sayfa", target_dim=512, anchor_len=400, max_tokens=1000):
    """
    Sayfanın belirli bir bölümünü işle
    """
    print(f"  {section_name} işleniyor...")
    
    try:
        # Sorgu oluştur
        query = await build_page_query(filename,
                                     page=page_num,
                                     target_longest_image_dim=target_dim,
                                     target_anchor_text_len=anchor_len)
        
        query['model'] = 'allenai_olmocr-7b-0225-preview'
        query['max_tokens'] = max_tokens
        query['temperature'] = 0.1
        
        # OpenAI istemcisini oluştur
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        
        # Görüntü işleme hatası durumunda alternatif yöntem dene
        if "image_url" in str(query) and query.get("messages", []):
            try:
                response = client.chat.completions.create(**query)
            except Exception:
                print("Alternatif görüntü işleme yöntemi deneniyor...")
                response = await process_pdf_as_image(client, filename, page_num, max_tokens)
        else:
            response = client.chat.completions.create(**query)
        
        if not response or not response.choices or not response.choices[0].message.content:
            print(f"API boş yanıt döndürdü")
            return None
            
        content = response.choices[0].message.content
        
        # JSON yanıtı ayrıştır
        try:
            model_obj = json.loads(content)
            if "natural_text" not in model_obj:
                model_obj["natural_text"] = content
        except json.JSONDecodeError:
            model_obj = {"primary_language": "tr", "natural_text": content}
        
        # PageResponse oluştur
        page_response = CustomPageResponse(
            primary_language=model_obj.get("primary_language", "tr"),
            natural_text=model_obj.get("natural_text", content)
        )

        result = PageResult(
            filename,
            page_num,
            page_response,
            input_tokens=getattr(response.usage, 'prompt_tokens', 0),
            output_tokens=getattr(response.usage, 'completion_tokens', 0),
            is_fallback=False,
        )
        
        print(f"  {section_name} tamamlandı - {len(page_response.natural_text)} karakter")
        return result
    except Exception as e:
        print(f"Sayfa bölümü işleme hatası ({section_name}): {e}")
        return None

def combine_texts(results):
    """
    Farklı bölümlerden gelen metinleri birleştir ve tekrarları temizle
    """
    if not results:
        return "Metin bulunamadı"
    
    # Tüm metinleri topla
    texts = [r.response.natural_text for r in results if hasattr(r.response, 'natural_text') and r.response.natural_text]
    
    # Geçerli metinleri filtrele
    valid_texts = [text for text in texts if text and len(text.strip()) > 0]
    
    if not valid_texts:
        return "İşlenebilir metin bulunamadı"
    
    # En uzun metni bul
    longest_text = max(valid_texts, key=len)
    
    # Diğer metinlerde olup en uzun metinde olmayan benzersiz cümleleri topla
    unique_sentences = set()
    all_sentences = set()
    
    # Önce en uzun metinden cümleleri topla
    for sentence in longest_text.replace('\n', ' ').split('. '):
        clean_sentence = sentence.strip()
        if clean_sentence:
            all_sentences.add(clean_sentence)
    
    # Diğer metinlerden benzersiz cümleleri topla
    for text in valid_texts:
        if text == longest_text:
            continue
            
        for sentence in text.replace('\n', ' ').split('. '):
            clean_sentence = sentence.strip()
            if clean_sentence and clean_sentence not in all_sentences:
                unique_sentences.add(clean_sentence)
                all_sentences.add(clean_sentence)
    
    # Birleştirilmiş metin oluştur
    combined = longest_text
    
    # Benzersiz cümleleri ekle
    if unique_sentences:
        combined += "\n\n" + ". ".join(unique_sentences) + "."
        
    return combined

async def process_page_with_overlap(filename, page_num, overlap_percent=30):
    """
    Sayfayı örtüşen bölümlere ayırarak işle
    """
    print(f"Sayfa {page_num} örtüşen bölümlerle işleniyor...")
    
    # Sabit düşük anchor değeri
    anchor_len = 300
    
    # Sayfayı 4 bölüme ayır
    num_sections = 4
    section_results = []
    
    # Her bölüm için
    for i in range(num_sections):
        section_name = f"Bölüm {i+1}/{num_sections}"
        
        # Bölüm başlangıç ve bitiş noktalarını hesapla
        section_start = max(0, i * (100 / num_sections) - (0 if i == 0 else overlap_percent))
        section_end = min(100, (i + 1) * (100 / num_sections) + (0 if i == num_sections - 1 else overlap_percent))
        
        print(f"  {section_name} işleniyor... (Sayfa yüzdesi: {section_start:.1f}% - {section_end:.1f}%)")
        
        try:
            # Bölümü işle - çözünürlüğü bölüme göre ayarla
            target_dim = 1024 if i == 0 or i == num_sections - 1 else 768
            
            result = await process_page_section(
                filename, 
                page_num, 
                section_name=section_name, 
                target_dim=target_dim,
                anchor_len=anchor_len,
                max_tokens=1000
            )
            
            if result:
                section_results.append(result)
                print(f"  {section_name} başarıyla işlendi")
            else:
                print(f"  {section_name} işlenemedi")
                
            # API'ye biraz nefes aldır
            time.sleep(1)
            
        except Exception as e:
            print(f"  {section_name} işleme hatası: {e}")
    
    if not section_results:
        raise ValueError(f"Sayfa {page_num} için hiçbir bölüm işlenemedi")
    
    # Tüm bölümleri birleştir
    combined_text = combine_texts(section_results)
    
    # PageResponse oluştur
    page_response = CustomPageResponse(primary_language="tr", natural_text=combined_text)
    
    return PageResult(
        filename,
        page_num,
        page_response,
        input_tokens=sum(r.input_tokens for r in section_results),
        output_tokens=sum(r.output_tokens for r in section_results),
        is_fallback=False
    )

async def process_simple(filename, page_num):
    """
    Basit PDF metin çıkarma - doğrudan PyPDF kullanarak
    """
    try:
        print(f"Basit metin çıkarma deneniyor (sayfa {page_num})...")
        reader = PdfReader(filename)
        
        if page_num > len(reader.pages):
            print(f"Hata: PDF dosyası sadece {len(reader.pages)} sayfa içeriyor")
            return None
            
        page = reader.pages[page_num-1]
        text = page.extract_text()
        
        if text and len(text.strip()) > 50:
            print(f"Basit metin çıkarma başarılı: {len(text)} karakter")
            
            page_response = CustomPageResponse(primary_language="tr", natural_text=text)
            
            return PageResult(
                filename,
                page_num,
                page_response,
                input_tokens=0,
                output_tokens=0,
                is_fallback=True,
            )
        else:
            print("Basit metin çıkarma başarısız veya çok az metin bulundu")
            return None
    except Exception as e:
        print(f"Basit metin çıkarma hatası: {e}")
        return None

async def process_fallback(filename, page_num):
    """
    Son çare işleme - düşük parametrelerle
    """
    print("Son çare işleme deneniyor...")
    try:
        # Düşük parametrelerle işle
        result = await process_page_section(
            filename, 
            page_num, 
            section_name="Son çare işleme", 
            target_dim=128, 
            anchor_len=100,
            max_tokens=800
        )
        return result
    except Exception as e:
        print(f"Son çare işleme hatası: {e}")
        
        # Doğrudan görüntü olarak işle
        try:
            client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
            response = await process_pdf_as_image(client, filename, page_num, max_tokens=1500)
            
            if response and response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content
                page_response = CustomPageResponse(primary_language="tr", natural_text=content)
                
                return PageResult(
                    filename,
                    page_num,
                    page_response,
                    input_tokens=getattr(response.usage, 'prompt_tokens', 0),
                    output_tokens=getattr(response.usage, 'completion_tokens', 0),
                    is_fallback=True,
                )
            else:
                print("Son çare görüntü işleme başarısız oldu")
                return None
        except Exception as e:
            print(f"Son çare görüntü işleme hatası: {e}")
            return None

# Ana fonksiyon
async def main():
    filename = "tahsilat-makbuzlari.pdf"
    
    # PDF dosyasını aç ve sayfa sayısını al
    reader = PdfReader(filename)
    num_pages = len(reader.pages)
    
    print(f"\n=== PDF dosyası toplam {num_pages} sayfa içeriyor ===")
    
    # İlk çalıştırmada çıktı dosyasını temizle
    with open("cikti.txt", "w", encoding="utf-8") as f:
        f.write("PDF İŞLEME SONUÇLARI\n")
        f.write("===================\n")
    
    all_results = []
    
    # Tüm sayfaları işle
    for page_num in range(1, num_pages + 1):
        print(f"\n=== Sayfa {page_num}/{num_pages} işleniyor ===")
        result = None
        
        # İşleme stratejilerini sırayla dene
        strategies = [
            ("Basit metin çıkarma", process_simple),
            ("Örtüşen bölümlerle işleme", process_page_with_overlap),
            ("Son çare işleme", process_fallback)
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                print(f"{strategy_name} deneniyor...")
                result = await strategy_func(filename, page_num)
                
                # Eğer sonuç başarılıysa ve yeterli metin varsa, döngüden çık
                if result and len(result.response.natural_text) > 200:
                    print(f"{strategy_name} başarılı!")
                    break
                else:
                    print(f"{strategy_name} yetersiz sonuç verdi, bir sonraki strateji deneniyor...")
            except Exception as e:
                print(f"{strategy_name} başarısız: {e}")
        
        # Sayfa sonucunu yazdır
        if result:
            print(f"\nSayfa {page_num} sonucu:")
            print("------------------------")
            print(result.response.natural_text[:500] + "..." if len(result.response.natural_text) > 500 else result.response.natural_text)
            print("------------------------\n")
            
            # Sonucu topla
            all_results.append(result)
            
            # Sonucu dosyaya ekle
            with open("cikti.txt", "a", encoding="utf-8") as f:
                f.write(f"\n\n--- SAYFA {page_num} ---\n\n")
                f.write(result.response.natural_text)
        else:
            print(f"Sayfa {page_num} için hiçbir işleme yöntemi başarılı olmadı.")
            with open("cikti.txt", "a", encoding="utf-8") as f:
                f.write(f"\n\n--- SAYFA {page_num} ---\n\n")
                f.write("İşleme başarısız oldu, metin elde edilemedi.")
    
    # İşlem tamamlandı mesajı
    print(f"\n=== Toplam {num_pages} sayfa işlendi ===")
    print(f"Sonuçlar 'cikti.txt' dosyasına kaydedildi.")
    
    return all_results

# Çalıştırma kodu
if __name__ == "__main__":
    import asyncio
    results = asyncio.run(main())