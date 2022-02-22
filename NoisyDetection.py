from utils import *
from options import Options

def main():
    # see options.py
    
    args = Options()
    GenerateEnvironment(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #TODO: move transform
    #TODO: experiment better transform to avoid overfitting
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    #TODO: create dataset based on arg
    train_dataset = ShuffledDataset('CIFAR10', './data', args.shuffle_percentage, train=True, download=True, transform=train_transform)
    cifar10_test_data = ShuffledDataset('CIFAR10', './data', 0, train=False, transform=test_transform)

    #TODO: scheduler
    test_loader = DataLoader(cifar10_test_data, args.batch_size, shuffle=False)
    
    # Pretrain to get the minimum epochs for model to converge
    model = torch.hub.load('pytorch/vision:v0.10.0', args.model, pretrained=False).to(device)
    finish_epochs = train(model, args.epochs, True, train_dataset, test_loader, device, args)
    
    # Begin noise detection
    model = torch.hub.load('pytorch/vision:v0.10.0', args.model, pretrained=False).to(device)
    loss_recording = train(model, finish_epochs, False, train_dataset, test_loader, device, args)

    # Report the noise detection results
    changed_indices = train_dataset.get_shuffle_mapping().keys()
    pred_indices = [t[1] for t in sorted(zip(loss_recording, range(len(train_dataset))), reverse=True, key=lambda x: x[0])[:len(changed_indices)]]
    noise_detected = list(set(changed_indices) & set(pred_indices))
    print("The model detected {} wrongly labeled training samples out of {} total samples".format(len(noise_detected), len(changed_indices)))

    # accuracy = eval(model, test_loader, device)
    # print(accuracy)

if __name__ == '__main__':
    main()