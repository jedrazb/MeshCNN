from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
from tqdm import tqdm


def run_test(epoch=-1):
    print('Dumping the embeddings')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()

    all_data = []

    for data in tqdm(dataset, total=len(dataset)//model.opt.batch_size):
        model.set_input(data)
        embeddings = model.dump_embeddings()
        labels = [l.item() for l in model.labels]
        eids = model.eids
        for idx in range(min(model.opt.batch_size, len(data['label']))):
            entry = (eids[idx], labels[idx], embeddings[idx].tolist())
            all_data.append(entry)

    data_csv = []
    for d in all_data:
        data_csv.append(
            '{};{};{}'.format(
                str(d[0]),
                str(d[1]),
                str(d[2])
            )
        )

    with open('dump.txt', 'w+') as f:
        content = '\n'.join(data_csv)
        f.write(content)

    print('Run completed')


if __name__ == '__main__':
    run_test()
