import os
import numpy as np
import xml.etree.ElementTree as ET
from dataset.data_util import pil_load_img
# from dataset.dataload import TextDataset, TextInstance
# from dataset.dataload_ocr import TextDataset_ocr, TextInstance

class OCR_data_xml(TextDataset_ocr):

    def __init__(self, root='./data/meter_read_online', is_training=True, transform=None):
        super().__init__(transform, is_training)
        self.dataset = []
        self.name = []
        image_path = f'{root}/img/'
        mask_path1 = f'{root}/anno'

        for image_name in os.listdir(image_path):
            mask_name = image_name.split('.')[0] + '.xml'
            self.dataset.append((f'{image_path}/{image_name}', f'{mask_path1}/{mask_name}'))
            self.name.append(image_name)

    @staticmethod
    def parse_txt(mask_path1):
        """
        XML file parser
        :param mask_path1: (str), XML file path
        :return: (list), TextInstance
        """

        polygons = []
        transcripts = []

        tree = ET.parse(mask_path1)
        root = tree.getroot()

        for obj in root.findall('kWh-rating'):
            points = []
            for pt in obj.findall('point'):
                x = float(pt.find('x').text)
                y = float(pt.find('y').text)
                points.append((x, y))

            label = obj.find('kWh-rating').text
            transcripts.append(label)

            points = np.array(points, dtype=np.int32)
            polygons.append(TextInstance(points, 'c', label))

        return polygons, transcripts

    def __getitem__(self, item):
        image_path, mask_path1 = self.dataset[item]
        idx = self.name[item]

        # Read image data
        image = pil_load_img(image_path)

        try:
            polygons, transcripts = self.parse_txt(mask_path1)
        except:
            polygons = None

        return self.get_training_data(image, polygons, transcripts, image_id=idx, image_path=image_path)

    def __len__(self):
        return len(self.dataset)


