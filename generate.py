import torch
import torchvision
import os
import argparse


from model import Generator
from utils import load_model, get_device

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate MNIST samples with the trained GAN.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for generation.")
    parser.add_argument("--checkpoint", type=str, default='checkpoints/G.pth',
                      help="Generator weights (default: checkpoints/G.pth, the submitted model).")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device.type}")

    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784

    model = Generator(g_output_dim=mnist_dim).to(device)
    model = load_model(model, args.checkpoint, device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    print('Model loaded.')



    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    n_samples = 0
    with torch.no_grad():
        while n_samples<10000:
            z = torch.randn(args.batch_size, 100).to(device)
            x = model(z)
            # tanh renvoie [-1, 1] ; on inverse Normalize(0.5, 0.5) pour revenir dans [0, 1] comme MNIST
            x = (x.reshape(args.batch_size, 28, 28) + 1) / 2
            for k in range(x.shape[0]):
                if n_samples<10000:
                    torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.png'))
                    n_samples += 1
