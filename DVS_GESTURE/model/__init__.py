from DVS_GESTURE.model.SNN import resnet_he
from DVS_GESTURE.model.ANN import resnet


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

    else:
        assert args.model in ['RESNET10', 'RESNET18', 'RESNET34', 'RESNET50',
                              'RESNET10-HE', 'RESNET18-HE', 'RESNET34-HE', 'RESNET68-HE', 'RESNET104-HE',]
        model = None

    return model
