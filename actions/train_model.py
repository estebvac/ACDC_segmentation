from torch.optim import Adam
from models.models import *
from actions.early_stopping import *
import os


def train_model(model, options, training_dataloader, validation_dataloader):
    # define the torch.device
    device = torch.device('cuda') if options['gpu_use'] else torch.device('cpu')
    # Unpack training hyperparameters
    lr = options['lr']
    pos_weight = torch.from_numpy(options['weights']).to(device).float()
    pos_weight = pos_weight / pos_weight.sum()
    # Model name:
    model_name = options['model_name']

    # send the model to the device
    model = model.to(device)

    # List to save the training results:
    train_loss_list = list()
    train_acc_list = list()
    val_loss_list = list()
    val_acc_list = list()

    earlystop = None
    if (options['patience'] != None):
        earlystop = EarlyStopping(patience=options['patience'], verbose=True)

    # training loop
    training = True
    epoch = 1

    optimizer = Adam(model.parameters(), lr=lr)

    ## DEFINE A LR SCHEDULER
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5,
                                                              verbose=True, threshold=0.0001, threshold_mode='rel',
                                                              cooldown=0, min_lr=0, eps=1e-08)

    try:
        # regenerate the training:
        model.load_state_dict(torch.load('checkpoint.pt'))
        print('Weights loaded')
    except:
        print('Weights not loaded.')

    try:
        while training:

            # epoch specific metrics
            train_loss = 0
            train_accuracy = 0
            val_loss = 0
            val_accuracy = 0
            train_dice = None
            val_dice = None

            # define the optimizer
            # optimizer = Adadelta(lesion_model.parameters())
            # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

            # -----------------------------
            # training samples
            # -----------------------------

            # set the model into train mode
            model.train()
            for number_batches_training, batch in enumerate(training_dataloader):
                # process batches: each batch is composed by training (x) and labels (y)
                # x = [32, 2, 32, 32, 32]
                # y = [32, 1, 32, 32, 32]

                x = batch[0].to(device)
                y = batch[1].to(device)

                # clear gradients
                optimizer.zero_grad()

                # infer the current batch
                pred = model(x)

                y_one_hot = torch.FloatTensor(x.size(0), 4, x.size(2), x.size(3), x.size(4))
                y_one_hot = y_one_hot.to(device)
                y_one_hot.scatter_(1, y.type(torch.LongTensor).to(device), 1)

                # compute the loss.
                # loss = tversky_loss(y.long() , pred, alpha, beta, pos_weight, eps=1e-7)
                loss = dice_loss(torch.log(torch.clamp(pred, 1E-7, 1.0)), y_one_hot)
                # print(pred)
                # loss = F.cross_entropy(torch.log(torch.clamp(pred, 1E-7, 1.0)),
                #                     y.squeeze(dim=1).long(), weight=pos_weight)

                train_loss += loss.item()

                # backward loss and next step
                loss.backward()
                optimizer.step()

                # compute the dice metric
                dice_score = calculate_dices_metric(4, pred, y, options)

                # compute the accuracy
                pred = pred.max(1, keepdim=True)[1]
                batch_accuracy = pred.eq(y.view_as(pred).long())
                train_accuracy += (batch_accuracy.sum().item() / np.prod(y.shape))
                if train_dice is None:
                    train_dice = dice_score
                else:
                    train_dice += dice_score

            # -----------------------------
            # validation samples
            # -----------------------------

            # set the model into train mode
            model.eval()
            for number_batches_validation, batch in enumerate(validation_dataloader):

                x = batch[0].to(device)
                y = batch[1].to(device)

                y_one_hot = torch.FloatTensor(x.size(0), 4, x.size(2), x.size(3), x.size(4))
                y_one_hot = y_one_hot.to(device)
                y_one_hot.scatter_(1, y.type(torch.LongTensor).to(device), 1)

                # infer the current batch
                with torch.no_grad():
                    pred = model(x)

                    # compute the loss.
                    # we ignore the index=2
                    # loss = tversky_loss(y.long() , pred, alpha, beta, pos_weight, eps=1e-7)
                    loss = dice_loss(torch.log(torch.clamp(pred, 1E-7, 1.0)), y_one_hot)
                    # loss = F.cross_entropy(torch.log(torch.clamp(pred, 1E-7, 1.0)),
                    #                      y.squeeze(dim=1).long(), weight=pos_weight)

                    val_loss += loss.item()

                    # compute the dice metric
                    dice_score = calculate_dices_metric(4, pred, y, options)

                    # compute the accuracy
                    pred = pred.max(1, keepdim=True)[1]
                    batch_accuracy = pred.eq(y.view_as(pred).long())
                    val_accuracy += batch_accuracy.sum().item() / np.prod(y.shape)
                    if val_dice is None:
                        val_dice = dice_score
                    else:
                        val_dice += dice_score

            # compute mean metrics
            train_loss /= (number_batches_training + 1)
            train_accuracy /= (number_batches_training + 1)
            val_loss /= (number_batches_validation + 1)
            val_accuracy /= (number_batches_validation + 1)
            train_dice /= (number_batches_training + 1)
            val_dice /= (number_batches_validation + 1)

            print('Epoch {:d} train_loss {:.4f} train_acc {:.4f}'.format(
                epoch, train_loss,
                train_accuracy), 'val_loss {:.4f} val_acc {:.4f}'.format(
                val_loss, val_accuracy), 'train_dice', train_dice, 'val_dice', val_dice)

            # save the metrics
            train_loss_list.append(train_loss)
            train_acc_list.append(train_accuracy)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_accuracy)

            # CHECK EARLY STOP
            if (earlystop != None):
                earlystop(val_loss, model)

                if (earlystop.early_stop):
                    print("Early stopping")
                    # Load the model with the best parameters
                    model.load_state_dict(torch.load('./checkpoint.pt'))
                    training = False

            # CHECK SCHEDULER
            lr_scheduler.step(val_loss)

            # update epochs
            epoch += 1

            # save weights
            torch.save(model.state_dict(),
                       os.path.join(options['save_path'], model_name + str(epoch) + '.pt'))

            if epoch >= options['num_epochs']:
                training = False
    except KeyboardInterrupt:
        pass


# Modified fromrom https://github.com/rogertrullo/pytorch/blob/rogertrullo-dice_loss/torch/nn/functional.py#L708
def dice_loss(input, target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 5, "Input must be a 5D Tensor."
    # uniques=np.unique(target.numpy())
    # assert set(list(uniques))<=set([0,1]), "target must only contain zeros and ones"

    probs = F.softmax(input, dim=1)
    num = probs * target  # b,c,h,w--p*g
    num = torch.sum(num, dim=4)  # b,c,h
    num = torch.sum(num, dim=3)
    num = torch.sum(num, dim=2)

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=4)  # b,c,h
    den1 = torch.sum(den1, dim=3)
    den1 = torch.sum(den1, dim=2)

    den2 = target * target  # --g^2
    den2 = torch.sum(den2, dim=4)  # b,c,h
    den2 = torch.sum(den2, dim=3)  # b,c
    den2 = torch.sum(den2, dim=2)  # b,c

    dice = 2 * (num / (den1 + den2))
    dice_eso = dice[:, 0:]

    dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz

    return dice_total


def tversky_loss(true, logits, alpha, beta, cl_weights=1, eps=1e-7):
    """Computes the Tversky loss [1].
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
        eps: added to the denominator for numerical stability.
    Returns:
        tversky_loss: the Tversky loss.
    Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff
    References:
        [1]: https://arxiv.org/abs/1706.05721
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 4, 1, 2, 3).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    fps = torch.sum(probas * (1 - true_1_hot), dims)
    fns = torch.sum((1 - probas) * true_1_hot, dims)
    num = intersection
    denom = intersection + (alpha * fps) + (beta * fns)

    tversky_loss = (num / (denom + eps))  # .mean()
    cl_weights = cl_weights / torch.sum(cl_weights)
    balanced_tversky_loss = tversky_loss * cl_weights
    return (1 - balanced_tversky_loss.sum())


def dice_metric(volume_counter, mask_counter):
    """
    Calcuate the dice index of one region of interest
    Parameters
    ----------
    volume_counter:     segmentation result
    mask_counter :      ground truth image

    Returns
    -------
    Dice score of the region of interest
    """
    num = 2 * (volume_counter.float() * mask_counter.float()).sum()
    den = volume_counter.sum() + mask_counter.sum()
    dice_tissue = num.div(den.float())
    return dice_tissue


def calculate_dices_metric(tissues: int, volume, gt, options, cl_weights=1) -> dict:
    """
    Calculates the dice score of all the regions segmented
    Parameters
    ----------
    tissues:        Number of regions to analize
    volume:         Segmentation result
    gt:             Ground truth image

    Returns
    -------
    The dice score of all the regions segmented.

    """
    device = torch.device('cuda') if options['gpu_use'] else torch.device('cpu')
    dices_per_tissue = torch.zeros([tissues, ])
    _, volume = torch.max(volume, 1)

    gt = gt.squeeze(1)

    for tissue_id in range(0, tissues):
        volume_counter = 1 * (volume == tissue_id)
        mask_counter = 1 * (gt == tissue_id)
        dice_tissue = dice_metric(volume_counter, mask_counter)
        dices_per_tissue[tissue_id - 1] = dice_tissue

    dices_per_tissue = dices_per_tissue.to(device)
    weigted_dice = dices_per_tissue * cl_weights
    return weigted_dice
