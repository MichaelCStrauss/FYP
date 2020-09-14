from fyp.data.flickr8k import Flickr8kDataModule


def hash_tensor(t):
    b = []
    for i in t[0].reshape((-1,))[0:100]:
        b.append(float(i))
    for i in t[-1].reshape((-1,))[0:100]:
        b.append(float(i))
    return hash(tuple(b))


def test_datamodule():
    datamodule = Flickr8kDataModule()
    datamodule.prepare_data()
    datamodule.setup()

    dataloader = datamodule.train_dataloader()

    images, captions, length, target = iter(dataloader).next()

    for caption, length, target in zip(
        captions.split(1), length.split(1), target.split(1)
    ):
        print(length)
        print(datamodule.encoder.decode(caption.squeeze(0)))
        print(datamodule.encoder.decode([target.squeeze(0)]))

    assert hash_tensor(images) == 6453947208640630704
    assert hash_tensor(captions) == -4859957097191475580
    assert hash_tensor(target) == -995428953964473571
