import pandas as pd
import time
import os
import json

from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.pytorch_data_loader import *
from src.pytorch_cnn_models import *
from src.utils import calc_f1_score, get_key
from config.models_config import image_size, num_epochs, batch_size, learning_rate

from torchsampler import ImbalancedDatasetSampler


def train(cnn, num_epochs, device, train_loader, optimizer, criterion, calc_f1_score, loss_stats, accuracy_stats, model_name):
    #TRAIN
    start_time = time.time()
    
    for epoch in tqdm(range(1, num_epochs + 1)):
        print ("Starting Epoch {}".format(epoch))
        # Loss and f1s score within the epoch
        train_epoch_loss = 0
        train_epoch_acc = 0
        
        cnn.train()
        
        i = 0
        
        for images, labels, image_row in train_loader:
            images = Variable(images).to(device)
            labels =  Variable(labels).to(device)
            
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(images)
            
            train_loss = criterion(outputs, labels)
            train_acc = calc_f1_score(outputs, labels)
            
            train_loss.backward()
            optimizer.step()
            
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
            
            i += 1
            
            if i % 100 == 0:
                print('Batch {0} Loss: {1} '.format(i, train_loss.item()))
        
        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        
        train_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        train_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        
        train_acc_df.to_csv(r'intermediate_outputs/train_acc_df.csv')
        train_loss_df.to_csv(r'intermediate_outputs/train_loss_df.csv')
        
        print(f'Epoch {epoch+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Training F1 Score: {train_epoch_acc/len(train_loader):.5f}')
        print("Epoch Done")
    
    print("Saving model at model/" + model_name)
    torch.save(cnn,"model/" + model_name + ".pt")
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
def test(test_loader):
    print("Testing..")
    y_pred_list = []
    image_file_name_list = []
    with torch.no_grad():
        cnn.eval()
        for test_images, test_labels, image_row, in tqdm(test_loader):
            
            images = Variable(test_images).to(device)    
            outpots = cnn(images)

            y_pred_softmax = torch.log_softmax(outpots, dim = 1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

            y_pred_list.append(y_pred_tags.cpu().numpy())
            
            image_file_name_list.append(str(int(image_row.cpu().numpy().astype(np.float))))
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    #save predictions
    predictions = dict()    
    #get_predictions and row_number as integer
    predictions["target"] = dict(zip(image_file_name_list ,y_pred_list))
    
    with open("predictions.json", "w") as p:
        json.dump(predictions, p)
    
    
    
if __name__ == '__main__':
    if torch.has_cuda:
        device = 'cuda'
    elif torch.has_mps:
        device = 'mps'
    else:
        device = 'cpu'    
    
    if not os.path.exists('model'):
        os.makedirs('model')
    
    # ---------------------------------------------------------------- TRAIN ---------------------------------------------------------------- #
    
    train_labels_df = pd.read_pickle("intermediate_outputs/train_labels.pickle")
    
    train_df = customdataset(train_labels_df,image_size, set_type="train" ,transforms = transforms)
    train_loader = (DataLoader(train_df, batch_size = batch_size, shuffle=False, sampler=ImbalancedDatasetSampler(train_df)))
    
    #get model
    cnn = AlexNet()
    #model to device
    cnn.to(device)
    accuracy_stats = {'train': []}
    loss_stats = {'train': []}
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate)
    
    train(cnn,
          num_epochs,
          device,
          train_loader,
          optimizer,
          criterion,
          calc_f1_score,
          loss_stats,
          accuracy_stats,
          model_name = "alexnet_cnn_model")
    
    
# ---------------------------------------------------------------- TEST ---------------------------------------------------------------- #    
    
    cnn  = torch.load("model/alexnet_cnn_model.pt")
    test_labels_df = pd.read_pickle("intermediate_outputs/test_labels.pickle")
    print("test set length: %d" % len(test_labels_df))
    test_df = customdataset(test_labels_df, image_size,set_type = "test",transforms = transforms) 
    test_loader = (DataLoader(test_df, batch_size = 1, shuffle=False))
    test(test_loader)