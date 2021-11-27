#libs
import face_recognition
import numpy as np
from PIL import Image
from deepface import DeepFace as deepf


#classes
class FaceRegError(Exception):
    pass


class NoData(Exception):
    pass


class Face():
    def __init__(self, path='', face_encode=True, fromdata=False):
        '''
        Создание лица. На вход подаётся массив numpy или название файла
        '''

        if path:
            try:
                self.img = self._file2img(path)
                if face_encode:
                    self.face = face_recognition.face_encodings(self.img)[0]
                else:
                    self.face = False
            except:
                raise FaceRegError("Face registration error. The image is damaged or there is no face on it")
        elif fromdata:
            self.img = fromdata['img']
            if face_encode:
                self.face = fromdata['face']
            else:
                self.face = False
        else:
            raise NoData("Have not data for work")

    def _file2img(self, file, mode='RGB'):
        '''
        Стандартизация
        '''

        im = Image.open(file)
        if mode:
            im = im.convert(mode)
        return np.array(im)

    def getdata(self):
        '''
        Выгрузка данных
        '''

        return {'img':self.img,
                'face':self.face}

    def check(self, unface:np.array):
        '''
        Сравнение лица с другим. На вход подаётся другое лицо в формате массива
        '''

        if type(self.face) != bool:
            NoData("Have not data for work")
        res = face_recognition.compare_faces([self.face], unface)
        return res[0]

    def verify(self, img, model_name='VGG-Face'):
        '''
        Сравнение лица с другим. На вход подаётся путь к изображению лица
        '''
        res = deepf.verify(img1_path=self.img, img2_path=img, model_name='VGG-Face')
        return {"verifed": res['verified'],
                "score": 1 - res['distance']}

    def emotion(self):
        '''
        Определение эмоции
        '''

        res = deepf.analyze(self.img, actions=['emotion'])
        return res['emotion']

    def whois(self):
        '''
        Определение возраста, пола и рассы
        '''
        res = deepf.analyze(self.img, actions=['age','gender','race'])
        return {'age':res['age'],
                'gender':res['gender'],
                'race':res['race']}
                
