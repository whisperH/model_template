from tools import *
from ModelZoo.model_structure import *

def run():
    # 1. get model parameter
    parser = LightningNet.add_model_specific_args(model_name, dataset_name)
    model_para = parser.parse_args()

    # 2. load data
    test_dataset = datasets.MNIST(root='Dataset', train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

    # 2. start test
    model = LightningNet.load_from_checkpoint('/home/entropy/code/SizeMeasure/Logs/example_minist_logs/bestbest_ckpt_epoch_4_v0.ckpt')

    eval_loss = 0
    eval_acc = 0
    model.eval()  # 模型转化为评估模式
    for X, label in test_loader:
        X = X.view(-1, 784)
        X = Variable(X)
        label = Variable(label)
        testout = model(X)
        testloss = nn.CrossEntropyLoss()(testout, label)
        eval_loss += float(testloss)
        _, pred = testout.max(1)
        num_correct = (pred == label).sum()
        acc = int(num_correct) / X.shape[0]
        eval_acc += acc
    print("testlose: " + str(eval_loss / len(test_loader)))
    print("testaccuracy:" + str(eval_acc / len(test_loader)) + '\n')


if __name__ == '__main__':
    model_name = 'example'
    dataset_name = 'minist'
    run()
