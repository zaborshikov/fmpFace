#libs
import face_recognition
import numpy as np
from PIL import Image


#classes
class FaceRegError(Exception):
    None

class Face():

    def __init__(self, img):
        '''
        Создание лица. На вход подаётся массив numpy или название файла
        '''
        if type(img)==str:
            img = self.file2img(img)
        try:
            self.face = face_recognition.face_encodings(img)[0]
        except:
            raise FaceRegError("Face registration error. The image is damaged or there is no face on it")

    def check(self, unface:np.array):
        '''
        Сравнение лица с другим. На вход подаётся другое лицо в формате массива
        '''
        result = face_recognition.compare_faces([self.face], unface)
        return result[0]

    def file2img(self, file, mode='RGB'):
        '''
        Приводит файл к формату массива
        '''
        im = Image.open(file)
        if mode:
            im = im.convert(mode)
        return np.array(im)
