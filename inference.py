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
        print('Cannot find input path: {0}'.format(image_path))
        exit()
    if not os.path.exists(model_path):
        print('Cannot find model path: {0}'.format(model_path))
        exit()

    ref_size = 512

    # Get x_scale_factor & y_scale_factor to resize image
    def get_scale_factor(im_h, im_w, ref_size):

        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32

        x_scale_factor = im_rw / im_w
        y_scale_factor = im_rh / im_h

        return x_scale_factor, y_scale_factor

    ##############################################
    #  Main Inference part
    ##############################################

    # read image
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # unify image channels to 3
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    # normalize values to scale it between -1 to 1
    im = (im - 127.5) / 127.5

    im_h, im_w, im_c = im.shape
    x, y = get_scale_factor(im_h, im_w, ref_size)

    # resize image
    im = cv2.resize(im, None, fx = x, fy = y, interpolation = cv2.INTER_AREA)

    # prepare input shape
    im = np.transpose(im)
    im = np.swapaxes(im, 1, 2)
    im = np.expand_dims(im, axis = 0).astype('float32')

    # Initialize session and get prediction
    session = onnxruntime.InferenceSession(model_path, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: im})

    # refine matte
    matte = (np.squeeze(result[0]) * 255).astype('uint8')
    matte = cv2.resize(matte, dsize=(im_w, im_h), interpolation = cv2.INTER_AREA)

    cv2.imwrite(output_path, matte)

    ##############################################
    # Optional - save png image without background
    ##############################################

    im_PIL = Image.open(image_path)
    matte = Image.fromarray(matte)
    im_PIL.putalpha(matte)   # add alpha channel to keep transparency
    im_PIL.save('static/uploads/detected.png')

    def resize_image(input_path, novo_tamanho, by_width):
        image = Image.open(input_path)

        width, height = image.size
        if by_width:
            new_width = novo_tamanho
            new_height = int((novo_tamanho / width) * height)
        else:
            new_width = int((novo_tamanho / height) * width)
            new_height = novo_tamanho

        resized = image.resize((new_width, new_height))

        resized.save(input_path)


    def adjust_images(images):
        open_images = [Image.open(image) for image in images]

        image_base_height, image_base_width = open_images[0].size
        image_sup_height, image_sup_width = open_images[1].size


        if image_sup_width > image_base_width:
            resize_image(images[1], image_base_width, False)
        if image_sup_height > image_base_height:
            resize_image(images[0], image_sup_height, True)


    def merge_images(images):
        open_images = [Image.open(image).convert("RGBA") for image in images]

        max_width_images = max(image.width for image in open_images)
        max_height_images = max(image.height for image in open_images)

        result = Image.new('RGBA', (max_width_images, max_height_images), (0, 0, 0, 0))

        image_base_width, image_base_height = open_images[0].size

        x_center = (image_base_width - open_images[1].width) // 2
        y_center = (image_base_height - open_images[1].height)


        result.paste(open_images[0], (0, 0), open_images[0])
        result.paste(open_images[1], (x_center, y_center), open_images[1])

        result.save(exit_path)

    image_base_path = "static/logo/MODnet.png"
    image_sub_path = "static/uploads/detected.png"
    exit_path = "static/uploads/saida.png"    
    images = [image_base_path, image_sub_path]

    adjust_images(images)
    merge_images(images)
    