import torch.nn as nn

def modify_model(model, args):
    
    if args.ground_color_space == 'L':
        model.ground_embedding.feature_extractor.extract_features.conv1_1 = \
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1))
        print('Ground embedding network modified for grayscale image')
    if args.aerial_color_space == 'L':
        model.aerial_embedding.feature_extractor.extract_features.conv1_1 = \
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1))
        print('Aerial embedding network modified for grayscale image')
    
    return model