#libs
import face_recognition
import numpy as np
from PIL import Image
from deepface import DeepFace as deepf


#classes
class FaceRegError(Exception):
    None


class Face():
    def __init__(self, path, face_encode=True):
        '''
        Создание лица. На вход подаётся массив numpy или название файла
        '''
        self.path = path
        try:
            self.img = self.file2img(path)
            if face_encode:
                self.face = face_recognition.face_encodings(self.img)[0]
        except:
            raise FaceRegError("Face registration error. The image is damaged or there is no face on it")

    def check(self, unface:np.array):
        '''
        Сравнение лица с другим. На вход подаётся другое лицо в формате массива
        '''
        res = face_recognition.compare_faces([self.face], unface)
        return res

    def verify(self, img, model_name='VGG-Face'):
        '''
        Сравнение лица с другим. На вход подаётся путь к изображению лица
        '''
        res = deepf.verify(img1_path=self.img, img2_path=img, model_name='VGG-Face')
        return 1 - res['distance']

    def file2img(self, file, mode='RGB'):
        '''
        Стандартизация
        '''
        im = Image.open(file)
        if mode:
            im = im.convert(mode)
        return np.array(im)
    def emotion(self):
        '''
        Определение эмоции
        '''
        res = deepf.analyze(self.img, actions=['emotion'])
        return res['emotion']
    def whois(self):
        res = deepf.analyze(self.img, actions=['age','gender','race'])
        return {'age':res['age'], 'gender':res['gender'], 'race':res['race']}
