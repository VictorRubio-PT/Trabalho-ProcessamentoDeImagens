
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# FUNÇÕES AUXILIARES


def carregar_imagem(caminho, cinza=False):
    """Carrega uma imagem usando OpenCV apenas para entrada."""
    modo = cv2.IMREAD_GRAYSCALE if cinza else cv2.IMREAD_COLOR
    img = cv2.imread(caminho, modo)
    if img is None:
        raise FileNotFoundError(f"Não foi possível carregar: {caminho}")
    if not cinza:
        # OpenCV lê em BGR; convertemos para RGB para exibição correta.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def salvar_rgb(caminho, img_rgb):
    """Salva uma imagem RGB convertendo para BGR antes do cv2.imwrite."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(caminho, img_bgr)


def mostrar(nome, img, cmap=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap=cmap)
    plt.title(nome)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def mostrar_lado_a_lado(imagens, titulos, cmaps=None):
    n = len(imagens)
    plt.figure(figsize=(5 * n, 5))
    for i, img in enumerate(imagens):
        plt.subplot(1, n, i + 1)
        plt.imshow(img, cmap=None if cmaps is None else cmaps[i])
        plt.title(titulos[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()



# QUESTÃO 1


def q1():
    img = carregar_imagem("Lenna.png")

    altura, largura = img.shape[:2]
    canais = 1 if img.ndim == 2 else img.shape[2]

    print("\n--- QUESTÃO 1 ---")
    print("Dimensões:", largura, "x", altura)
    print("Canais:", canais)

    # 1) Reamostragem para uma matriz 3 vezes menor.
    
    nova_altura = max(1, altura // 3)
    nova_largura = max(1, largura // 3)

    # Implementação manual por Nearest Neighbor.
    reduzida = resize_nearest(img, nova_largura, nova_altura)
    mostrar("Lenna - matriz 3 vezes menor", reduzida)

    # 2) Reamostragem para 25 DPI.

    # Como PNGs nem sempre possuem DPI confiável, primeiro tentamos
    # obter a informação pelo Pillow.
    with Image.open("Lenna.png") as im:
        dpi = im.info.get("dpi", None)

    print("DPI encontrado no arquivo:", dpi)

    # DPI de referência assumido quando o arquivo não informa nenhum
    # (96 DPI é o padrão típico de exibição em monitores no Windows,
    # que é exatamente o contexto citado no enunciado).
    DPI_REFERENCIA = 96

    if dpi is not None and dpi[0] > 0:
        dpi_original = dpi[0]
        print("Usando DPI do próprio arquivo como referência.")
    else:
        dpi_original = DPI_REFERENCIA
        print(f"Arquivo não informou DPI. Assumindo DPI de referência = {DPI_REFERENCIA}.")

    fator = 25 / dpi_original
    nova_w = max(1, round(largura * fator))
    nova_h = max(1, round(altura * fator))

    img_25dpi = resize_nearest(img, nova_w, nova_h)

    # Salva já com o pixel reamostrado e o metadado de 25 DPI.
    pil_img = Image.fromarray(img_25dpi)
    pil_img.save("Lenna_25dpi.png", dpi=(25, 25))

    print(f"Imagem 25 DPI criada: {nova_w} x {nova_h} (a partir de referência de {dpi_original} DPI)")

    print("\nAnálise:")
    print("- A redução para 1/3 já diminui bastante a quantidade de pixels;")
    print("  detalhes finos do rosto e da pena do chapéu começam a se perder,")
    print("  e bordas antes suaves ficam mais serrilhadas (efeito do Nearest Neighbor).")
    print("- DPI relaciona a quantidade de pixels ao tamanho físico de impressão/exibição:")
    print("  quanto menor o DPI para o mesmo tamanho físico, menos pixels são necessários.")
    print(f"- Como {dpi_original} -> 25 DPI é uma redução de fator {fator:.3f}, a imagem final")
    print("  fica bem menor e com perda visível de detalhe, mas ainda reconhecível,")
    print("  pois a estrutura geral (contraste, formas) se mantém mesmo com poucos pixels.")
    print("- Como o PNG original não trazia DPI, assumimos 96 DPI (padrão de tela) como")
    print("  referência; se outro valor de referência for usado, o tamanho final muda.")



# QUESTÃO 2 - QUANTIZAÇÃO DE 1 A 8 BITS


def quantizar(img, bits):
    """
    Reduz a quantidade de níveis de intensidade.
    8 bits -> 256 níveis
    1 bit  -> 2 níveis
    """
    niveis = 2 ** bits
    passo = 256 / niveis

    # Quantização uniforme:
    quant = np.floor(img.astype(np.float64) / passo) * passo

    # Representação visual usando o centro aproximado de cada intervalo.
    quant += passo / 2

    # np.round (e não apenas astype, que trunca) garante que o
    # arredondamento para uint8 seja feito corretamente.
    quant = np.clip(np.round(quant), 0, 255)

    return quant.astype(np.uint8)


def q2():
    img = carregar_imagem("Lenna.png")

    print("\n--- QUESTÃO 2 ---")

    imagens = []
    titulos = []

    for bits in range(1, 9):
        q = quantizar(img, bits)
        imagens.append(q)
        titulos.append(f"{bits} bits - {2**bits} níveis")

        salvar_rgb(f"Lenna_{bits}bits.png", q)

    plt.figure(figsize=(16, 8))
    for i, img_q in enumerate(imagens):
        plt.subplot(2, 4, i + 1)
        plt.imshow(img_q)
        plt.title(titulos[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("Comparação:")
    print("- 1 bit: apenas 2 níveis, aparência muito simplificada.")
    print("- 2 bits: 4 níveis.")
    print("- 3 bits: 8 níveis.")
    print("- ...")
    print("- 8 bits: 256 níveis, praticamente mantém a faixa original de intensidade.")
    print("- Quanto menor a profundidade, maior a perda de informação tonal.")



# QUESTÃO 3 - DEGRADÊ EM ESCALA DE CINZA 8 BITS


def q3():
    print("\n--- QUESTÃO 3 ---")

    # Matriz 256x256.
    # Cada coluna recebe valores de 0 a 255.
    gradiente = np.zeros((256, 256), dtype=np.uint8)

    for x in range(256):
        gradiente[:, x] = x

    cv2.imwrite("degrade_256x256.png", gradiente)

    mostrar("Degradê de cinza - 8 bits", gradiente, cmap="gray")

    print("A imagem possui 256 níveis possíveis de cinza.")
    print("0 representa preto e 255 representa branco.")



# QUESTÃO 4 - CORES RGB E MISTURAS


def q4():
    print("\n--- QUESTÃO 4 ---")

    tamanho = 300

    vermelho = np.zeros((tamanho, tamanho, 3), dtype=np.uint8)
    verde = np.zeros((tamanho, tamanho, 3), dtype=np.uint8)
    azul = np.zeros((tamanho, tamanho, 3), dtype=np.uint8)

    # RGB = [R, G, B]
    vermelho[:, :, 0] = 255
    verde[:, :, 1] = 255
    azul[:, :, 2] = 255

    # Misturas aditivas.
    ciano = vermelho * 0 + verde + azul
    magenta = vermelho + azul
    amarelo = vermelho + verde
    branco = vermelho + verde + azul

    # Como arrays uint8 podem estourar ao somar, fazemos em int
    # e depois limitamos para 255.
    ciano = np.clip(verde.astype(int) + azul.astype(int), 0, 255).astype(np.uint8)
    magenta = np.clip(vermelho.astype(int) + azul.astype(int), 0, 255).astype(np.uint8)
    amarelo = np.clip(vermelho.astype(int) + verde.astype(int), 0, 255).astype(np.uint8)
    branco = np.clip(
        vermelho.astype(int) + verde.astype(int) + azul.astype(int),
        0, 255
    ).astype(np.uint8)

    imagens = [vermelho, verde, azul, ciano, magenta, amarelo, branco]
    nomes = ["Vermelho", "Verde", "Azul", "Ciano", "Magenta", "Amarelo", "Branco"]

    for img, nome in zip(imagens, nomes):
        salvar_rgb(f"{nome}.png", img)

    plt.figure(figsize=(14, 8))
    for i, (img, nome) in enumerate(zip(imagens, nomes)):
        plt.subplot(2, 4, i + 1)
        plt.imshow(img)
        plt.title(nome)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    print("Explicação:")
    print("- RGB é um modelo de cor aditivo.")
    print("- Vermelho + Verde = Amarelo.")
    print("- Vermelho + Azul = Magenta.")
    print("- Verde + Azul = Ciano.")
    print("- Vermelho + Verde + Azul = Branco.")



# QUESTÃO 5 - HSV, RGB E CMYK


def q5():
    print("\n--- QUESTÃO 5 ---")
    print("""
RGB:
- Red, Green, Blue.
- Modelo aditivo.
- Muito usado em telas, câmeras e imagens digitais.

HSV:
- Hue (matiz), Saturation (saturação), Value (valor/brilho).
- Separa melhor características perceptuais da cor.
- Útil para seleção, segmentação e manipulação de cores.

CMYK:
- Cyan, Magenta, Yellow, Key/Black.
- Modelo subtrativo.
- Muito usado em impressão.
- As tintas absorvem parte da luz refletida.
""")



# QUESTÃO 6


def q6():
    # Você pode trocar "Lenna.png" por qualquer outra imagem.
    img = carregar_imagem("Lenna.png")

    print("\n--- QUESTÃO 6 ---")

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    # Exibição dos canais individualmente.
    mostrar_lado_a_lado(
        [R, G, B],
        ["Canal R", "Canal G", "Canal B"],
        ["gray", "gray", "gray"]
    )

    # Mantendo apenas os pares solicitados.
    somente_RB = np.zeros_like(img)
    somente_RB[:, :, 0] = R
    somente_RB[:, :, 2] = B

    somente_RG = np.zeros_like(img)
    somente_RG[:, :, 0] = R
    somente_RG[:, :, 1] = G

    somente_BG = np.zeros_like(img)
    somente_BG[:, :, 1] = G
    somente_BG[:, :, 2] = B

    mostrar_lado_a_lado(
        [somente_RB, somente_RG, somente_BG],
        ["R + B", "R + G", "B + G"]
    )

    salvar_rgb("canal_R.png", np.stack([R, R, R], axis=2))
    salvar_rgb("canal_G.png", np.stack([G, G, G], axis=2))
    salvar_rgb("canal_B.png", np.stack([B, B, B], axis=2))

    salvar_rgb("R+B.png", somente_RB)
    salvar_rgb("R+G.png", somente_RG)
    salvar_rgb("B+G.png", somente_BG)

    # Conversão para HSV é uma das conversões autorizadas pelo enunciado.
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    mostrar_lado_a_lado(
        [H, S, V],
        ["Hue (H)", "Saturation (S)", "Value (V)"],
        ["gray", "gray", "gray"]
    )

    # Para facilitar a visualização do H, S e V, também salvamos cada
    # canal normalizado em 0-255.
    H_vis = cv2.normalize(H, None, 0, 255, cv2.NORM_MINMAX)
    S_vis = cv2.normalize(S, None, 0, 255, cv2.NORM_MINMAX)
    V_vis = cv2.normalize(V, None, 0, 255, cv2.NORM_MINMAX)

    cv2.imwrite("HSV_H.png", H_vis)
    cv2.imwrite("HSV_S.png", S_vis)
    cv2.imwrite("HSV_V.png", V_vis)

    print("Resultados:")
    print("- R, G e B mostram quanto cada componente participa de cada pixel.")
    print("- R+B remove o componente verde.")
    print("- R+G remove o componente azul.")
    print("- B+G remove o componente vermelho.")
    print("- H representa a matiz.")
    print("- S representa a saturação.")
    print("- V representa o valor/brilho.")



# FUNÇÕES MANUAIS PARA QUESTÃO 7


def resize_nearest(img, nova_largura, nova_altura):
    """
    Redimensionamento manual usando Nearest Neighbor.
    Não usa cv2.resize.
    """
    altura, largura = img.shape[:2]

    if img.ndim == 2:
        saida = np.zeros((nova_altura, nova_largura), dtype=img.dtype)
    else:
        saida = np.zeros((nova_altura, nova_largura, img.shape[2]), dtype=img.dtype)

    for y in range(nova_altura):
        origem_y = min(int(y * altura / nova_altura), altura - 1)

        for x in range(nova_largura):
            origem_x = min(int(x * largura / nova_largura), largura - 1)
            saida[y, x] = img[origem_y, origem_x]

    return saida


def translacao(img, tx, ty):
    """Translação manual."""
    h, w = img.shape[:2]

    saida = np.zeros_like(img)

    for y in range(h):
        for x in range(w):
            novo_x = x + tx
            novo_y = y + ty

            if 0 <= novo_x < w and 0 <= novo_y < h:
                saida[novo_y, novo_x] = img[y, x]

    return saida


def rotacao_manual(img, angulo_graus):
    """Rotação manual em torno do centro da imagem."""
    h, w = img.shape[:2]
    saida = np.zeros_like(img)

    ang = np.deg2rad(angulo_graus)
    cos_a = np.cos(ang)
    sin_a = np.sin(ang)

    cx = (w - 1) / 2
    cy = (h - 1) / 2

    for y2 in range(h):
        for x2 in range(w):
            # Mapeamento inverso: destino -> origem.
            dx = x2 - cx
            dy = y2 - cy

            x = cos_a * dx + sin_a * dy + cx
            y = -sin_a * dx + cos_a * dy + cy

            xi = round(x)
            yi = round(y)

            if 0 <= xi < w and 0 <= yi < h:
                saida[y2, x2] = img[yi, xi]

    return saida


def bilinear_pixel(img, x, y):
    """Calcula um pixel por interpolação bilinear."""
    h, w = img.shape[:2]

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)

    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))

    dx = x - x0
    dy = y - y0

    p00 = img[y0, x0].astype(float)
    p10 = img[y0, x1].astype(float)
    p01 = img[y1, x0].astype(float)
    p11 = img[y1, x1].astype(float)

    top = p00 * (1 - dx) + p10 * dx
    bottom = p01 * (1 - dx) + p11 * dx

    return top * (1 - dy) + bottom * dy


def resize_bilinear(img, nova_largura, nova_altura):
    """Redimensionamento manual usando interpolação bilinear."""
    h, w = img.shape[:2]

    if img.ndim == 2:
        saida = np.zeros((nova_altura, nova_largura), dtype=np.uint8)
    else:
        saida = np.zeros((nova_altura, nova_largura, img.shape[2]), dtype=np.uint8)

    for y2 in range(nova_altura):
        y = y2 * (h - 1) / max(1, nova_altura - 1)

        for x2 in range(nova_largura):
            x = x2 * (w - 1) / max(1, nova_largura - 1)
            valor = bilinear_pixel(img, x, y)
            saida[y2, x2] = np.clip(valor, 0, 255).astype(np.uint8)

    return saida


def q7():
    img = carregar_imagem("L-RES.png")

    print("\n--- QUESTÃO 7 ---")

    h, w = img.shape[:2]

    # 1) Translação.
    # Altere estes valores se o professor tiver especificado outro deslocamento.
    trans = translacao(img, tx=50, ty=30)

    # 2) Redimensionamento sem interpolação.
    nova_w = max(1, int(w * 1.5))
    nova_h = max(1, int(h * 1.5))
    redim = resize_nearest(img, nova_w, nova_h)

    # 3) Rotação.
    rot = rotacao_manual(img, 30)

    mostrar_lado_a_lado(
        [img, trans, redim, rot],
        ["Original", "Translação", "Redimensionada", "Rotação"]
    )

    salvar_rgb("LRES_translacao.png", trans)
    salvar_rgb("LRES_redimensionada_nearest.png", redim)
    salvar_rgb("LRES_rotacao.png", rot)

    # O enunciado permite função pronta para Bicúbica.
    # Para comparação, usamos a imagem redimensionada e rotacionada
    # como base das interpolações.
    base = rot

    nn = resize_nearest(base, w, h)
    bilinear = resize_bilinear(base, w, h)

    # Bicúbica autorizada pelo enunciado.
    base_bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
    bicubica_bgr = cv2.resize(base_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
    bicubica = cv2.cvtColor(bicubica_bgr, cv2.COLOR_BGR2RGB)

    mostrar_lado_a_lado(
        [nn, bilinear, bicubica],
        ["Nearest Neighbor", "Bilinear", "Bicúbica"]
    )

    salvar_rgb("LRES_nearest.png", nn)
    salvar_rgb("LRES_bilinear.png", bilinear)
    salvar_rgb("LRES_bicubica.png", bicubica)

    print("Comparação:")
    print("- Nearest Neighbor escolhe o pixel mais próximo; é simples e rápido.")
    print("- Bilinear considera os pixels vizinhos e produz transições mais suaves.")
    print("- Bicúbica considera uma vizinhança maior e tende a produzir resultado mais suave e detalhado.")
    print("- A diferença fica mais perceptível em bordas, diagonais e detalhes finos.")



# QUESTÃO 8 - HISTOGRAMA E EQUALIZAÇÃO


def histograma_manual(img):
    """Calcula histograma de uma imagem em tons de cinza sem cv2.calcHist."""
    hist = np.zeros(256, dtype=np.int64)

    for valor in img.flatten():
        hist[int(valor)] += 1

    return hist


def equalizar_histograma_manual(img):
    """
    Equalização de histograma manual:
    1. histograma
    2. histograma acumulado
    3. CDF normalizada
    4. nova intensidade para cada pixel
    """
    hist = histograma_manual(img)

    cdf = np.cumsum(hist)

   
    cdf_min = cdf[cdf > 0][0]
    total_pixels = img.size

    tabela = np.zeros(256, dtype=np.uint8)

    for i in range(256):
        if cdf[i] >= cdf_min:
            novo_valor = round(
                (cdf[i] - cdf_min) / (total_pixels - cdf_min) * 255
            )
            tabela[i] = np.clip(novo_valor, 0, 255)

    equalizada = tabela[img]
    return equalizada, hist


def q8():
    print("\n--- QUESTÃO 8 ---")

    img = carregar_imagem("Contrast.png", cinza=True)

    hist_original = histograma_manual(img)

    plt.figure(figsize=(8, 5))
    plt.bar(range(256), hist_original, width=1)
    plt.title("Histograma - imagem original")
    plt.xlabel("Intensidade")
    plt.ylabel("Quantidade de pixels")
    plt.tight_layout()
    plt.show()

    equalizada, _ = equalizar_histograma_manual(img)

    hist_equalizado = histograma_manual(equalizada)

    mostrar_lado_a_lado(
        [img, equalizada],
        ["Original", "Equalizada"],
        ["gray", "gray"]
    )

    plt.figure(figsize=(8, 5))
    plt.bar(range(256), hist_equalizado, width=1)
    plt.title("Histograma - imagem equalizada")
    plt.xlabel("Intensidade")
    plt.ylabel("Quantidade de pixels")
    plt.tight_layout()
    plt.show()

    cv2.imwrite("CONTRAST_equalizada.png", equalizada)

    print("Interpretação:")
    print("- O histograma mostra quantos pixels existem em cada nível de intensidade.")
    print("- Se os valores estiverem concentrados em uma faixa estreita, o contraste pode ser limitado.")
    print("- A equalização usa a distribuição acumulada para espalhar melhor as intensidades.")
    print("- O objetivo é aumentar a utilização da faixa dinâmica e melhorar o contraste global.")



# MENU PRINCIPAL


def menu():
    while True:
        print("\n==========================================")
        print(" LISTA AVALIATIVA 1 - PDI")
        print("==========================================")
        print("1 - Questão 1")
        print("2 - Questão 2")
        print("3 - Questão 3")
        print("4 - Questão 4")
        print("5 - Questão 5")
        print("6 - Questão 6")
        print("7 - Questão 7")
        print("8 - Questão 8")
        print("0 - Executar todas")
        print("9 - Sair")

        opcao = input("Escolha: ").strip()

        try:
            if opcao == "1":
                q1()
            elif opcao == "2":
                q2()
            elif opcao == "3":
                q3()
            elif opcao == "4":
                q4()
            elif opcao == "5":
                q5()
            elif opcao == "6":
                q6()
            elif opcao == "7":
                q7()
            elif opcao == "8":
                q8()
            elif opcao == "0":
                q1()
                q2()
                q3()
                q4()
                q5()
                q6()
                q7()
                q8()
            elif opcao == "9":
                print("Encerrando...")
                break
            else:
                print("Opção inválida.")
        except Exception as erro:
            print("\nERRO:", erro)
            print("Verifique se as imagens necessárias estão na mesma pasta do programa.")


if __name__ == "__main__":
    menu()
