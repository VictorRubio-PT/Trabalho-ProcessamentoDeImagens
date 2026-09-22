Trabalho-ProcessamentoDeImagens

Disciplina: Processamento Digital de Imagens Professor: Marcio Salmazo Ramos Aluno: Victor Rubio

Sobre o trabalho

Lista Avaliativa 1 de Processamento Digital de Imagens, implementada em Python. Todos os algoritmos de manipulação de imagem (redimensionamento, quantização, translação, rotação, interpolação e equalização de histograma) foram implementados manualmente, sem uso de funções prontas do OpenCV, exceto para conversões de espaço de cor (RGB ↔ HSV, RGB ↔ BGR).

Questões resolvidas
Carregamento de imagem, informações (dimensões/canais), reamostragem para matriz 3x menor e reamostragem para 25 DPI.
Requantização da imagem para profundidades de 1 a 8 bits.
Geração de imagem artificial com degradê de cinza (256x256, 8 bits).
Criação de imagens de cores primárias (R, G, B) e suas misturas aditivas (Ciano, Magenta, Amarelo, Branco).
Comparação entre os espaços de cor RGB, HSV e CMYK.
Separação de canais RGB e HSV, e combinações de canais (R+G, R+B, B+G).
Translação, redimensionamento e rotação manuais, com comparação entre os métodos de interpolação Nearest Neighbor, Bilinear e Bicúbica.
Histograma e equalização de histograma (implementação manual).
Como executar
1. Instalar as dependências
bash
pip install opencv-python numpy matplotlib pillow
2. Rodar o script
bash
python lista_pdi.py

Um menu interativo será exibido no terminal, permitindo escolher qual questão executar (ou rodar todas de uma vez com a opção 0).

Arquivos
Arquivo	Descrição
lista_pdi.py	Código-fonte com a resolução de todas as questões
Lenna.png	Imagem usada nas questões 1, 2 e 6
L-RES.png	Imagem usada na questão 7 (transformações geométricas)
Contrast.png	Imagem usada na questão 8 (histograma/equalização)
