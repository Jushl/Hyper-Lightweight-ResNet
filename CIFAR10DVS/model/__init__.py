from CIFAR10DVS.model.SNN import resnet_he
from CIFAR10DVS.model.ANN import resnet


def build_model(args):
    if args.model == 'RESNET10':
        model = resnet.resnet10(num_cls=args.num_classes, channel=args.channel)
    elif args.model == 'RESNET18':
        model = resnet.resnet18(num_cls=args.num_classes, channel=args.channel)
    elif args.model == 'RESNET34':
        model = resnet.resnet34(num_cls=args.num_classes, channel=args.channel)
    elif args.model == 'RESNET50':
        model = resnet.resnet50(num_cls=args.num_classes, channel=args.channel)

    elif args.model == 'RESNET10-HE':
        model = resnet_he.resnet10(num_cls=args.num_classes, channel=args.channel)
    elif args.model == 'RESNET18-HE':
        model = resnet_he.resnet18(num_cls=args.num_classes, channel=args.channel)
    elif args.model == 'RESNET34-HE':
        model = resnet_he.resnet34(num_cls=args.num_classes, channel=args.channel)
    elif args.model == 'RESNET68-HE':
        model = resnet_he.resnet68(num_cls=args.num_classes, channel=args.channel)
    elif args.model == 'RESNET104-HE':
        model = resnet_he.resnet104(num_cls=args.num_classes, channel=args.channel)

    else:
        assert args.model in ['RESNET10', 'RESNET18', 'RESNET34', 'RESNET50',
                              'RESNET10-HE', 'RESNET18-HE', 'RESNET34-HE', 'RESNET68-HE', 'RESNET104-HE',
                              ]
        model = None

    return model
