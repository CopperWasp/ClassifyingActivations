import torch
import torchvision
import cnn
import torchvision.transforms as transforms
import random
import resnet as r
import parameters as p
import time
from argparse import ArgumentParser
torch.backends.cudnn.deterministic = True


parser = ArgumentParser()
parser.add_argument("-b", "--batch", dest="batch_size", help="Specify batch size.", metavar="int")
parser.add_argument("-t", "--type",dest="net_type", help="resnet or cnn.")
parser.add_argument("-m", "--tsim", dest="sim_threshold", help="Similarity threshold.")
parser.add_argument("-p", "--tsk", dest="skip_threshold", help="Layer switch threshold.")
parser.add_argument("-d", "--dev", dest="device", help="cpu or cuda:0.")
args = parser.parse_args()


p.batch_size = int(args.batch_size)
p.net_type = args.net_type
p.sim_threshold = float(args.sim_threshold)
p.skip_threshold = float(args.skip_threshold)
p.device = args.device


random.seed(1)
torch.manual_seed(1)

batch_size = p.batch_size

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device(p.device)



if p.net_type == 'cnn':
    net = cnn.Net()
    net.load_state_dict(torch.load('./checkpoint/ckpt.t7', map_location='cpu'))
    net.to(device)

if p.net_type == 'resnet':
    state = torch.load('./checkpoint/resnet.pth')
    net = r.resnet()
    net.load_state_dict(state['net'])
    net.eval()
    net.to(device)


# test
start = time.time()
correct = 0
total = 0
counter = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device) # gpu
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
end = time.time()
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
print('Num. instances skipped: '+str(net.skip_counter))
print('Elapsed time (s): '+str(end-start))


directory = './experiments/' + str(p.net_type) + "/"
name = str(p.batch_size) + str(p.sim_threshold) + str(p.num_layers) + str(p.skip_threshold) + ".txt"
with open(directory+name, 'w') as f:
    print("Parameter Setting", file=f)
    print("\tBatch Size = "+str(p.batch_size), file=f)
    print("\tNet Type = "+str(p.net_type), file=f)
    print("\tSimilarity Th. = "+str(p.sim_threshold), file=f)
    print("\tSkip Th. = "+str(p.skip_threshold), file=f)
    print("\tDevice = "+str(p.device), file=f)
    print("", file=f)
    print("Results",file=f)
    print('\tElapsed time (s): '+str(end-start), file=f)
    print('\tAccuracy of the network on the 10000 test images: %d %%' % (100 * correct / total), file=f)
    print('\t# Instances skipped: '+str(net.skip_counter), file=f)
    print("\tInput Counts = "+str(net.t_act.input_counts), file=f)
    print("\tSimilar Counts = "+str(net.t_act.similar_counts), file=f)
    print("\tClass Layers = "+str(net.t_act.class_layers), file=f)

print("Done, check "+str(directory+name)+" for results.")
