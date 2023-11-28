import os
import cv2
import numpy as np
import onnxruntime
from PIL import Image
import datetime

global x
x = datetime.datetime.now()

model_path = "pretrained/modnet.onnx"
output_path = f"static/results/output_{x}.png"


def predict(image_path):
    if not os.path.exists(image_path):
        print(f'Não foi possivel encontrar o caminho do input: {image_path}')
        exit()
    if not os.path.exists(model_path):
        print(f'Não foi possivel encontrar o caminho do modelo: {model_path}')
        exit()
    ref_size = 512


    def get_scale_factor(im_height, im_width, ref_size):

        if max(im_height, im_width) < ref_size or min(im_height, im_width) > ref_size:
            if im_width >= im_height:
                im_rh = ref_size
                im_rw = int(im_width / im_height * ref_size)
            elif im_width < im_height:
                im_rw = ref_size
                im_rh = int(im_height / im_width * ref_size)
        else:
            im_rh = im_height
            im_rw = im_width

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32

        x_scale_factor = im_rw / im_width
        y_scale_factor = im_rh / im_height

        return x_scale_factor, y_scale_factor

    #Lê a imagem usando OpenCV e converte de BGR para RGB.
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    #Normaliza os valores de pixel para o intervalo [-1, 1]
    im = (im - 127.5) / 127.5

    im_heigth, im_width, im_c = im.shape
    x, y = get_scale_factor(im_heigth, im_width, ref_size)
    im = cv2.resize(im, None, fx = x, fy = y, interpolation = cv2.INTER_AREA)

    #Faz a imagem ser transposta
    im = np.transpose(im)
    #Troca os eixos da imagem
    im = np.swapaxes(im, 1, 2)
    #Adiciona uma nova dimensão à imagem no eixo = 0
    im = np.expand_dims(im, axis = 0).astype('float32')

    #Cria uma sessão de inferencia
    session = onnxruntime.InferenceSession(model_path, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: im})

    #Converte o resultado da inferência para o formato de imagem e salva a saída.
    matte = (np.squeeze(result[0]) * 255).astype('uint8')
    matte = cv2.resize(matte, dsize=(im_width, im_heigth), interpolation = cv2.INTER_AREA)

    cv2.imwrite(output_path, matte)

    im_PIL = Image.open(image_path)
    #Converte um array NumPy em um objeto de imagem
    matte = Image.fromarray(matte)
    #Adiciona o canal alfa(transparencia)
    im_PIL.putalpha(matte)
    im_PIL.save('static/uploads/detected.png')

    #Define os tamanhos da imagem para fazer a sobreposição de forma correta
    def resize_image(image_path, new_size, by_width):
        im = Image.open(image_path)

        width, height = im.size
        if by_width:
            new_width = new_size
            new_heigth = int((new_size / width) * height)
        else:
            new_width = int((new_size / height) * width)
            new_heigth = new_size

        resized = im.resize((new_width, new_heigth))

        resized.save(image_path)

    #Função para sobreposição da imagem recortada no novo bakcground
    def adjust_images(imgs):
        open_im = [Image.open(image) for image in imgs]

        im_base_h, im_base_w = open_im[0].size
        im_sup_h, im_sup_w = open_im[1].size

        if im_sup_w > im_base_w:
            resize_image(imgs[1], im_base_w, False)
        if im_sup_h > im_base_h:
            resize_image(imgs[0], im_sup_h, True)

    #Junção da imagem recortada no seu novo background
    def merge_images(imgs):
        open_im = [Image.open(img).convert("RGBA") for img in imgs]

        max_w_img = max(img.width for img in open_im)
        max_h_im = max(img.height for img in open_im)

        result = Image.new('RGBA', (max_w_img, max_h_im), (0, 0, 0, 0))

        im_base_w, im_base_h = open_im[0].size

        x_center = (im_base_w - open_im[1].width) // 2
        y_center = (im_base_h - open_im[1].height)

        result.paste(open_im[0], (0, 0), open_im[0])
        result.paste(open_im[1], (x_center, y_center), open_im[1])

        result.save(exit_path)


    im_base_path = "static/uploads/background.png"
    im_sub_path = "static/uploads/detected.png"
    exit_path = "static/uploads/saida.png"    
    imgs = [im_base_path, im_sub_path]

    adjust_images(imgs)
    merge_images(imgs)
    