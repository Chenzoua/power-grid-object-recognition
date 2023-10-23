import torch
from PIL import Image
import numpy as np

# from config import common_config as config
from .modelcrnn import CRNN
from .ctc_decoder import ctc_decode
from .datasetcrnn import Synth90kDataset


def recognize_text(image, decode_method='beam_search', beam_size=10):
    # 加载模型
    model_path = '././Model_all/crnn.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_height = 32
    img_width = 100
    num_class = len(Synth90kDataset.LABEL2CHAR) + 1
    crnn = CRNN(1, img_height, img_width, num_class,
                map_to_seq_hidden=64,
                rnn_hidden=256,
                leaky_relu=False)
    crnn.load_state_dict(torch.load(model_path, map_location=device))
    crnn.to(device)
    crnn.eval()

    # 转换图像数据为Tensor并发送到设备
    image = torch.from_numpy(image).float()
    image = image.unsqueeze(0)  # 添加 batch 维度
    image = image.unsqueeze(0)  # 添加通道维度
    image = image.to(device)


    # 进行预测
    with torch.no_grad():
        logits = crnn(image)
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)
        preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
                           label2char=Synth90kDataset.LABEL2CHAR)

    return ''.join(preds[0])  # 返回预测的文本结果

if __name__ == '__main__':
    image_path = '../demo/200.jpg'  # 替换为要识别的图像路径
    # model_path = 'checkpoints/crnn.pt'  # 替换为训练好的模型的路径

    result = recognize_text(image_path)
    print(f'Recognized text: {result}')
