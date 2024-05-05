import timeit
from collections import OrderedDict

import torch
from torchvision import transforms, datasets

from CNN_submission import CNN

torch.multiprocessing.set_sharing_strategy('file_system')


def compute_score(acc, acc_thresh):
    min_thres, max_thres = acc_thresh
    if acc <= min_thres:
        base_score = 0.0
    elif acc >= max_thres:
        base_score = 100.0
    else:
        base_score = float(acc - min_thres) / (max_thres - min_thres) \
                     * 100
    return base_score

def test(
        model,
        dataset_name,
        device,

):
    if dataset_name == "MNIST":
        test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))]))

    elif dataset_name == "CIFAR10":
        test_dataset = datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    else:
        raise AssertionError(f"Invalid dataset: {dataset_name}")

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False)

    model.eval()
    num_correct = 0
    total = 0
    for batch_idx, (data, targets) in enumerate(test_loader):
        data = data.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            output = model(data)
            predicted = torch.argmax(output, dim=1)
            total += targets.size(0)
            num_correct += (predicted == targets).sum().item()

    acc = float(num_correct) / total
    return acc

class Args:
    """
    command-line arguments
    """

    """    
    'MNIST': run on MNIST dataset 
    'CIFAR10': run on CIFAR10 dataset 
    """
    dataset = "MNIST"
    # dataset = "CIFAR10"

    """
    set to 0 to run on cpu
    """
    gpu = 1

def main():
    args = Args()
    try:
        import paramparse
        paramparse.process(args)
    except ImportError:
        pass

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    acc_thresh = dict(
        MNIST=(0.93, 0.97),
        CIFAR10=(0.55, 0.65),
    )

    
    start = timeit.default_timer()
    results = CNN(args.dataset, device)
    model = results['model']

    stop = timeit.default_timer()
    run_time = stop - start

    accuracy = test(
        model,
        args.dataset,
        device,
    )

    score = compute_score(accuracy, acc_thresh[args.dataset])
    result = OrderedDict(
        accuracy=accuracy,
        score=score,
        run_time=run_time
    )
    print(f"result on {args.dataset}:")
    for key in result:
        print(f"\t{key}: {result[key]}")



if __name__ == "__main__":
    main()