import numpy as np


def convertListofList(lst):
    return [[el] for el in lst]


def split_dataset(img_dataframe, valid_size, test_size):
    num_train = len(img_dataframe)
    indices = list(range(num_train))
    split = int(np.floor((valid_size + test_size) * num_train))

    # Define a random seed
    np.random.seed(42)
    np.random.shuffle(indices)

    # Split the index
    train_idx, rest_idx = indices[split:], indices[:split]
    split_val = int(np.floor((valid_size / (valid_size + test_size)) * split))
    val_idx, test_idx = rest_idx[:split_val], rest_idx[split_val:]

    train_dataframe = img_dataframe.iloc[train_idx]
    val_dataframe = img_dataframe.iloc[val_idx]
    test_dataframe = img_dataframe.iloc[test_idx]
    print('Training samples: ', len(train_dataframe), ' , Validation samples: ', len(val_dataframe),
          ' , Test samples: ', len(test_dataframe))

    input_dictionary = {}

    input_dictionary['input_train_data'] = dict(zip(train_dataframe['Folder'].values.tolist() +
                                                    train_dataframe['Folder'].values.tolist(),
                                                    convertListofList(train_dataframe['dias'].values.tolist() +
                                                                      train_dataframe['sist'].values.tolist())))

    input_dictionary['input_train_labels'] = dict(zip(train_dataframe['Folder'].values.tolist() +
                                                      train_dataframe['Folder'].values.tolist(),
                                                      convertListofList(train_dataframe['dias_gt'].values.tolist() +
                                                                        train_dataframe['sist_gt'].values.tolist())))

    input_dictionary['input_train_rois'] = dict(zip(train_dataframe['Folder'].values.tolist() +
                                                    train_dataframe['Folder'].values.tolist(),
                                                    convertListofList(train_dataframe['ROI'].values.tolist() +
                                                                      train_dataframe['ROI'].values.tolist())))

    input_dictionary['input_val_data'] = dict(zip(val_dataframe['Folder'].values.tolist() +
                                                  val_dataframe['Folder'].values.tolist(),
                                                  convertListofList(val_dataframe['dias'].values.tolist() +
                                                                    val_dataframe['sist'].values.tolist())))

    input_dictionary['input_val_labels'] = dict(zip(val_dataframe['Folder'].values.tolist() +
                                                    val_dataframe['Folder'].values.tolist(),
                                                    convertListofList(val_dataframe['dias_gt'].values.tolist() +
                                                                      val_dataframe['sist_gt'].values.tolist())))

    input_dictionary['input_val_rois'] = dict(zip(val_dataframe['Folder'].values.tolist() +
                                                  val_dataframe['Folder'].values.tolist(),
                                                  convertListofList(val_dataframe['ROI'].values.tolist() +
                                                                    val_dataframe['ROI'].values.tolist())))

    input_dictionary['input_test_data'] = dict(zip(test_dataframe['Folder'].values.tolist() +
                                                   test_dataframe['Folder'].values.tolist(),
                                                   convertListofList(test_dataframe['dias'].values.tolist() +
                                                                     test_dataframe['sist'].values.tolist())))

    input_dictionary['input_test_labels'] = dict(zip(test_dataframe['Folder'].values.tolist() +
                                                     test_dataframe['Folder'].values.tolist(),
                                                     test_dataframe['dias_gt'].values.tolist() +
                                                     test_dataframe['sist_gt'].values.tolist()))

    input_dictionary['input_test_rois'] = dict(zip(test_dataframe['Folder'].values.tolist() +
                                                   test_dataframe['Folder'].values.tolist(),
                                                   test_dataframe['ROI'].values.tolist() +
                                                   test_dataframe['ROI'].values.tolist()))

    return input_dictionary
