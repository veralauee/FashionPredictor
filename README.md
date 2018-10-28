# Fashion-Predictor-Retriever
fashion predictor and retriever

=======

1. create a model to save pretrained ImageNet weigths
cd saved_models
download weights from 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
name it as 'vgg16_bn.pth'

cd ..

2. download dataset
cd dataset

first setup images
download DeepFashion dataset from https://drive.google.com/drive/folders/0B7EVK8r0v71pekpRNUlMS3Z5cUk?usp=sharing

then setup labels
download labels from https://drive.google.com/open?id=1EfLX2mGg7cFSqon7gxpgCaofnorZ5ZG2
unzip labels.zip

In such case, we have directory structures like dataset/Img and dataset/labels

3. train a fashion-attribute predictor (Weighted BCE loss)
python train_Predictor.py

4. test a fashion-attribute predictor
python Predictor.py

5. train a fashion-attribute retriever (Triplet loss)
python train_Retriever.py

6. test a fashion-attribute predictor
python Retriever.py

5. demo
python demo.py --line_num [line_num]

line_num represents a line in dataset/labels/test.txt, which is the path of an testing image.


=======

predict attributes of fashion items (done)
retrieve same fashion items (done)
retrieve compatible fashion sets (to do)

=============

If you have further question, please contact Xin Liu(xin.liu4@duke.edu), Ziwei Liu(lz013@ie.cuhk.edu.hk)
