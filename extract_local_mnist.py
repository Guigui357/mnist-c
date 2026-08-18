import os
import gzip
import struct

FILES = {
    "train_img": "train-images-idx3-ubyte.gz",
    "train_lbl": "train-labels-idx1-ubyte.gz",
    "test_img": "t10k-images-idx3-ubyte.gz",
    "test_lbl": "t10k-labels-idx1-ubyte.gz"
}

def extract_mnist_to_pgm(img_gzip, lbl_gzip, output_dir, max_samples):
    if not os.path.exists(img_gzip) or not os.path.exists(lbl_gzip):
        print(f"❌ Erro: Arquivos necessários não encontrados na pasta 'tmp'.")
        print(f"Certifique-se de que {os.path.basename(img_gzip)} e {os.path.basename(lbl_gzip)} estão lá dentro.")
        return False
        
    print(f"📦 Extraindo imagens para '{output_dir}'...")
    
    with gzip.open(img_gzip, 'rb') as f_img, gzip.open(lbl_gzip, 'rb') as f_lbl:
        # Lendo cabeçalhos binários
        magic_lbl, num_lbl = struct.unpack(">II", f_lbl.read(8))
        magic_img, num_img, rows, cols = struct.unpack(">IIII", f_img.read(16))
        
        if magic_lbl != 2049 or magic_img != 2051:
            print("❌ Erro: Formato interno de arquivo inválido ou corrompido.")
            return False

        limit = min(num_img, max_samples)
        digit_counters = {i: 0 for i in range(10)}

        for _ in range(limit):
            label = ord(f_lbl.read(1))
            img_data = f_img.read(rows * cols)
            
            target_dir = os.path.join(output_dir, str(label))
            os.makedirs(target_dir, exist_ok=True)
            
            file_index = digit_counters[label]
            file_path = os.path.join(target_dir, f"{file_index}.pgm")
            
            # Gravando no formato PGM binário (P5)
            with open(file_path, 'wb') as pgm:
                header = f"P5\n{cols} {rows}\n255\n".encode('ascii')
                pgm.write(header)
                pgm.write(img_data)
                
            digit_counters[label] += 1

    print(f"✅ Sucesso! {limit} imagens extraídas em: {output_dir}")
    return True

def main():
    print("📂 Verificando arquivos locais na pasta 'tmp'...")
    
    # 1. Extrair os dados de treino locais (limitado a 1000 amostras conforme seu TRAIN em C)
    success_train = extract_mnist_to_pgm(
        os.path.join("tmp", FILES["train_img"]),
        os.path.join("tmp", FILES["train_lbl"]),
        os.path.join("data", "train"),
        max_samples=10000
    )
    
    # 2. Extrair os dados de teste locais (limitado a 200 amostras conforme seu TEST em C)
    success_test = extract_mnist_to_pgm(
        os.path.join("tmp", FILES["test_img"]),
        os.path.join("tmp", FILES["test_lbl"]),
        os.path.join("data", "test"),
        max_samples=2000
    )
    
    if success_train and success_test:
        print("\n🎉 Tudo pronto! As pastas 'data/train' e 'data/test' foram preenchidas com imagens PGM reais.")
        print("Agora você já pode executar o código compilado em C.")

if __name__ == "__main__":
    main()
