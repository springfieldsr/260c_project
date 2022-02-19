from utils import *
from options import Options

def main():
    # see options.py
    
    args = Options()
    GenerateEnvironment(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #TODO: move transform
    resnet_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = ShuffledDataset('CIFAR10', './data', 0, train=True, download=True, transform=resnet_transform)
    cifar10_test_data = ShuffledDataset('CIFAR10', './data', 0, train=False, transform=resnet_transform)

    #TODO: scheduler
    test_loader = DataLoader(cifar10_test_data, args.batch_size, shuffle=False)
    
    # Import resnet18 pretrained model
    model = torch.hub.load('pytorch/vision:v0.10.0', args.model, pretrained=False).to(device)

    train(model, args.epochs, train_dataset, test_loader, device,args)
    accuracy = eval(model, test_loader, device)
    print(accuracy)

if __name__ == '__main__':
    main()