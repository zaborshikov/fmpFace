#libs
import face_recognition
import numpy as np
import dlib
from PIL import Image
from deepface import DeepFace as deepf


#classes
class FaceRegError(Exception):
    '''
    Класс ошибки решистрации лица
    '''
    pass


class Face():
    '''
    Класс содердащий данные о лице
    '''
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


#functions
def check(face_1, face_2, unface):
    '''
    Сравнение лица с другим с помощью библиотеки face_recognition
    '''

    res = face_recognition.compare_faces([face_1], face_2)
    return res[0]

def verify(img_1, img_2, model_name='VGG-Face'):
    '''
    Сравнение лица с другим с помощью библиотеки DeepFace
    '''

    # models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
    res = deepf.verify(img1_path=img_1, img2_path=img_2, model_name='VGG-Face')
    return {"verifed": res['verified'],
            "score": 1 - res['distance']}
