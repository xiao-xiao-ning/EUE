import random
import copy
import torch
from utils.models import resnet34, BiLSTMModel, TransformerModel
from utils.data_loader import load_data
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import argparse
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json

random.seed(29)
torch.set_num_threads(32)
torch.manual_seed(19)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, dataloader, num_classes):
    """
    Evaluate accuracy and F1 score for each class, and compute overall F1 score.
    
    Args:
        model (torch.nn.Module): Trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        num_classes (int): Number of classes in the dataset.
    
    Returns:
        dict: Per-class accuracy and F1-score, and overall F1-score.
    """
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for _, (data, labels) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)
            output = model(data)

            _, predicted = torch.max(output, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    # print(all_predictions)
    # exit()
    
    precision = precision_score(all_labels, all_predictions, average=None)
    recall = recall_score(all_labels, all_predictions, average=None)
    f1 = f1_score(all_labels, all_predictions, average=None)
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return f1, f1_macro, accuracy

if __name__ == "__main__":

    parser = argparse.ArgumentParser() 

    parser.add_argument('--dataset', type=str, default='', help="Dataset to train on")
    parser.add_argument('--architecture', type=str, default="", choices=['resnet', 'transformer', 'bilstm'])

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--savedir', type=str, default="classification_models")
    parser.add_argument('--inplanes', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=4)    
    
    #transformer and bi-lstm
    parser.add_argument('--use_transformer', type=bool, default=False)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--dim_feedforward', type=int, default=256)
    parser.add_argument('--dropout', type=int, default=0.2)
    parser.add_argument('--timesteps', type=int, default=1024) #128 140 1639
    
    args = parser.parse_args()
    print("Model::::::::::::", args.architecture)
    print("Data::::::::::::", args.dataset)

    model_root_dir = args.savedir
    model_dir = os.path.join(model_root_dir, args.dataset)
    os.makedirs(model_dir, exist_ok=True)

    
    train_dataset, val_dataset, test_dataset, _ = load_data(data_name=args.dataset)
    print('Dataset ...', 'Train Shape', len(train_dataset), 'Test Shape', len(test_dataset))
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


    if args.architecture == "resnet":
        if args.use_transformer:
            exit()
        net = resnet34(args, num_classes = args.num_classes).to(device)
    elif args.architecture == 'bilstm':
        net = BiLSTMModel(args, num_classes = args.num_classes).to(device) 
    elif args.architecture == 'transformer':
        net = TransformerModel(args, num_classes = args.num_classes).to(device)
    
    # if args.dataset == 'Earthquakes' and args.architecture == 'bilstm':
    #     args.lr = 0.02
    # if args.dataset == 'Earthquakes' and args.architecture == 'transformer':
    #     args.lr = 3e-4
    
    # if args.dataset == 'BME' and args.architecture == 'bilstm':
    #     args.lr = 0.02
        

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), args.lr)

    net = net.to(device)
    best_model = copy.deepcopy(net)
    best_f1_macro = -1
    score_json_file = os.path.join(model_dir, args.architecture+'_scores.json')
    for epoch in range(args.n_epochs):
        net.train()
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = net(batch_data)
            loss = criterion(outputs, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        net.eval()
        scores, f1_macro, accuracy = evaluate_model(model=net, dataloader=val_loader, num_classes=args.num_classes)
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{args.n_epochs}], Training Loss: {loss.item():.4f}, Validation Scores: {scores}, Accuracy: {accuracy}, F1 Macro: {f1_macro}")
        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            print("Save best model ...", best_f1_macro)
            torch.save(net.state_dict(), os.path.join(model_dir, args.architecture+'_'+'.pth'))
            best_model = copy.deepcopy(net)
            with open(score_json_file, "w") as json_file:
                json.dump(list(scores), json_file, indent=4)  # Use indent for pretty formatting
    scores, f1_macro, accuracy = evaluate_model(model=best_model, dataloader=test_loader, num_classes=args.num_classes)
    print(f'Accuracy::{accuracy}, F1 macro: {f1_macro}')
    with open(score_json_file, "w") as json_file:
        json.dump({'f1 score':list(scores), 'accuracy': accuracy, 'f1_macro': f1_macro}, json_file, indent=4)  # Use indent for pretty formatting

