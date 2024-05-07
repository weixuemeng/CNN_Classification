from tqdm import tqdm


import timeit
from FNN_submission import FNN

class Params:
    class BatchSize:
        train = 128
        val = 128
        test = 1000

    def __init__(self):
        self.device = 'gpu'
        self.loss_type = "ce"
        self.batch_size = Params.BatchSize()
        self.n_epochs = 10
        self.learning_rate = 1e-1
        self.momentum = 0.5


def get_dataloaders(batch_size):
    
    import torch
    from torch.utils.data import random_split
    import torchvision

    """

    :param Params.BatchSize batch_size:
    :return:
    """

    MNIST_training = torchvision.datasets.MNIST('.', train=True, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

    MNIST_test_set = torchvision.datasets.MNIST('.', train=False, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

    # create a training and a validation set
    MNIST_train_set, MNIST_val_set = random_split(MNIST_training, [55000, 5000])

    train_loader = torch.utils.data.DataLoader(MNIST_train_set, batch_size=batch_size.train, shuffle=True)

    val_loader = torch.utils.data.DataLoader(MNIST_val_set, batch_size=batch_size.val, shuffle= False)

    test_loader = torch.utils.data.DataLoader(MNIST_test_set,
                                              batch_size=batch_size.test, shuffle= False)

    return train_loader, val_loader, test_loader


def train(net, optimizer, train_loader, device):
    net.train()
    pbar = tqdm(train_loader, ncols=100, position=0, leave=True)
    avg_loss = 0
    for batch_idx, (data, target) in enumerate(pbar):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = net(data)
        loss = net.get_loss(output, target)
        loss.backward()
        optimizer.step()

        loss_sc = loss.item()

        avg_loss += (loss_sc - avg_loss) / (batch_idx + 1)

        pbar.set_description('train loss: {:.6f} avg loss: {:.6f}'.format(loss_sc, avg_loss))


def validation(net, validation_loader, device):
    net.eval()
    validation_loss = 0
    correct = 0
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        output = net(data)
        loss = net.get_loss(output, target)
        validation_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    validation_loss /= len(validation_loader.dataset)
    print('\nValidation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        validation_loss, correct, len(validation_loader.dataset),
        100. * correct / len(validation_loader.dataset)))


def test(net, test_loader, device):
    net.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)

        output = net(data)
        loss = net.get_loss(output, target)

        test_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    params = Params()

    try:
        import paramparse
    except ImportError:
        print("paramparse is unavailable so commandline arguments will not work")
    else:
        paramparse.process(params)

    import torch
    import torch.optim as optim
    
    import torch.nn.functional as F
    import torchvision

    random_seed = 1
    torch.manual_seed(random_seed)
    

    if params.device != 'cpu' and torch.cuda.is_available():
        device = torch.device("cuda")
        print('Running on GPU: {}'.format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device("cpu")
        print('Running on CPU')

    train_loader, val_loader, test_loader = get_dataloaders(params.batch_size)

    if params.loss_type == 'l2':
        loss_str = 'L2'
    elif params.loss_type == 'ce':
        loss_str = 'cross entropy'
    else:
        raise AssertionError(f'invalid loss type: {params.loss_type}')


    net = FNN(params.loss_type, 10).to(device)
    optimizer = optim.SGD(net.parameters(), lr=params.learning_rate,
                          momentum=params.momentum)

    print(f'\nusing implementation with {loss_str} loss\n')

    start = timeit.default_timer()

    with torch.no_grad():
        validation(net, val_loader, device)
    for epoch in range(params.n_epochs):
        print(f'\nepoch {epoch + 1} / {params.n_epochs}\n')
        train_start = timeit.default_timer()

        train(net, optimizer, train_loader, device)

        train_stop = timeit.default_timer()
        train_runtime = train_stop - train_start
        print(f'\ntrain runtime: {train_runtime:.2f} secs')

        with torch.no_grad():
            validation(net, val_loader, device)
    with torch.no_grad():
        test(net, test_loader, device)

    stop = timeit.default_timer()

    runtime = stop - start

    print(f'total runtime: {runtime:.2f} secs')


if __name__ == "__main__":
    main()
