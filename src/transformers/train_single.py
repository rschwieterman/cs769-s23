import torch
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Speech2TextForConditionalGeneration, Speech2TextProcessor
from transformers.s2tdataset import S2TDataset
from transformers.s2tmodel import CustomS2T
from IPython import embed

model = CustomS2T()
old_model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)
model.train()

if sys.argv[1] == "freeze":
    for param in model.model.model.encoder.parameters():
        param.requires_grad = False
    for param in model.model.model.decoder.parameters():
        param.requires_grad = False
    for param in model.model.lm_head.parameters():
        param.requires_grad = False

processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
feature_extractor = processor.feature_extractor
ds_train =load_dataset("librispeech_asr", "clean", split="train.100")
ds_validate =load_dataset("librispeech_asr", "clean", split="validation")

s2tData = S2TDataset(ds_train, None)
valData = S2TDataset(ds_validate, None)

bs = 20

train_dataloader = DataLoader(s2tData, shuffle=True, batch_size=bs,collate_fn=s2tData.collate_fn)
val_dataloader = DataLoader(valData, shuffle=False, batch_size=bs,collate_fn=s2tData.collate_fn)

def loss_func(pred,label,mask, keyword_pred, keyword_gt):
    logits = F.log_softmax(pred,dim=-1)
    bs = pred.shape[0]
    t_losses = torch.zeros(bs)
    keyword_gt = torch.tensor(keyword_gt, device='cuda', dtype=torch.float)
    for b in range(bs):
        t_losses[b] = torch.sum(mask[b] * F.cross_entropy(logits[b],label[b], reduction='none'))
    keyword_loss = F.binary_cross_entropy_with_logits(keyword_pred,keyword_gt)

    return (t_losses.sum()/bs) + float(sys.argv[2]) * keyword_loss


optimizer = optim.AdamW(model.parameters(), lr=1e-4)
print("begin training")
best_v_epoch_loss = 1e15
save_index = 0
for epoch in range(int(sys.argv[3])):
    epoch_loss = 0.0
    v_epoch_loss = 0.0
    correct = 0
    total = 0
    for batch_data in tqdm(train_dataloader):
        optimizer.zero_grad()
        outter2 = model.forward(batch_data[0].to(device), decoder_input_ids=batch_data[1].to(device), decoder_attention_mask=batch_data[2].to(device), output_attentions=True)
        keyword_label = batch_data[4]
        loss = loss_func(outter2.cls_logit_out, batch_data[3].to(device), batch_data[2].to(device),  outter2.key_detect,keyword_label )
        loss.backward()
        epoch_loss += loss.detach().cpu().item()
        pred = torch.argmax(outter2['key_detect'], dim=1)
        lab = torch.argmax(torch.tensor(batch_data[-1]), dim=1)
        correct += torch.sum(pred.detach().cpu() == lab)
        total += len(lab)
        optimizer.step()

    print("TRAIN Accuracy: ", correct / total)
    correct = 0
    total = 0
    with torch.no_grad():
        for val_data in tqdm(val_dataloader):
            optimizer.zero_grad()
            voutputs = model.forward(val_data[0].to(device).detach(), decoder_input_ids=val_data[1].to(device).detach(), decoder_attention_mask=val_data[2].to(device).detach())
            keyword_label = val_data[4]
            vloss = loss_func(voutputs.cls_logit_out, val_data[3].to(device), val_data[2].to(device),voutputs.key_detect,keyword_label)
            v_epoch_loss += vloss.detach().cpu().item()
            pred = torch.argmax(voutputs['key_detect'], dim=1)
            lab = torch.argmax(torch.tensor(val_data[-1]), dim=1)
            correct += torch.sum(pred.detach().cpu() == lab)
            total += len(lab)

    print("VAL Accuracy: ", correct / total)
    
    print(epoch_loss)
    print(epoch_loss/len(train_dataloader))
    print(v_epoch_loss/len(val_dataloader))
    if(v_epoch_loss < best_v_epoch_loss):
        torch.save(model.state_dict(), f"best_model_save_{sys.argv[1]}_{sys.argv[2]}.pyt")
        print("saving")
        best_v_epoch_loss = v_epoch_loss
