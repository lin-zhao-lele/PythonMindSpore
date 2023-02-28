from mindvision.dataset import Mnist
import numpy as np
from mindspore import Tensor
import matplotlib.pyplot as plt
from mindspore.train import Model
from mindvision.classification.models import lenet
import mindspore.nn as nn

mnist = Mnist("./mnist", split="train", batch_size=6, resize=32)
dataset_infer = mnist.run()
ds_test = dataset_infer.create_dict_iterator()
data = next(ds_test)
images = data["image"].asnumpy()
labels = data["label"].asnumpy()

plt.figure()
for i in range(1, 7):
    plt.subplot(2, 3, i)
    plt.imshow(images[i-1][0], interpolation="None", cmap="gray")
plt.show()

network = lenet(num_classes=10, pretrained=False)

# 定义损失函数
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

# 定义优化器函数
net_opt = nn.Momentum(network.trainable_params(), learning_rate=0.01, momentum=0.9)

# 使用函数model.predict预测image对应分类
model = Model(network, loss_fn=net_loss, optimizer=net_opt, metrics={'accuracy'})
output = model.predict(Tensor(data['image']))
predicted = np.argmax(output.asnumpy(), axis=1)


# 输出预测分类与实际分类
print(f'Predicted: "{predicted}", Actual: "{labels}"')