import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import os



def get_image_data():

    # 获取图像数据
    MNIST_dataset = MNIST(root='./FCNNModelCreater/MNIST/data', train=True, download=False, transform=transforms.ToTensor())
    image, label = MNIST_dataset[4] 
    image_pil = transforms.ToPILImage()(image) # 将图像从 Tensor 转换回 PIL 图像
    
    # 保存图像为 .png 文件
    save_path = './FCNNModelCreater'
    filename = f"image_{label}.png"  # 使用标签作为文件名
    image_pil.save(os.path.join(save_path, filename))
    

    print('当前图像标签为：', label)
    return image.squeeze().numpy().flatten()  # layer_1.input


