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
model.load_state_dict(torch.load(sys.argv[1]))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
feature_extractor = processor.feature_extractor
ds_validate = load_dataset("librispeech_asr", "clean", split="test")

valData = S2TDataset(ds_validate, None)

bs = 20

val_dataloader = DataLoader(valData, shuffle=False, batch_size=bs, collate_fn=valData.collate_fn)

def loss_func(pred,label,mask, keyword_pred, keyword_gt):
    logits = F.log_softmax(pred,dim=-1)
    bs = pred.shape[0]
    t_losses = torch.zeros(bs)
    keyword_gt = torch.tensor(keyword_gt, device='cuda', dtype=torch.float)
    for b in range(bs):
        t_losses[b] = torch.sum(mask[b] * F.cross_entropy(logits[b],label[b], reduction='none'))
    keyword_loss = F.binary_cross_entropy_with_logits(keyword_pred,keyword_gt)
    return (t_losses.sum()/bs), keyword_loss

save_index = 0
correct = 0
total = 0
l_epoch_loss = 0.0
k_epoch_loss = 0.0
with torch.no_grad():
    for val_data in tqdm(val_dataloader):
        voutputs = model.forward(val_data[0].to(device).detach(), decoder_input_ids=val_data[1].to(device).detach(),
                                 decoder_attention_mask=val_data[2].to(device).detach(), output_attentions=True)
        keyword_label = val_data[4]
        lloss, kloss = loss_func(voutputs.cls_logit_out, val_data[3].to(device), val_data[2].to(device),
                          voutputs.key_detect, keyword_label)
        l_epoch_loss += lloss.detach().cpu().item()
        k_epoch_loss += kloss.detach().cpu().item()
        pred = torch.argmax(voutputs['key_detect'], dim=1)
        lab = torch.argmax(torch.tensor(val_data[-1]), dim=1)
        correct += torch.sum(pred.detach().cpu() == lab)
        total += len(lab)

print("TEST Accuracy: ", correct / total)

print("SEQ LOSS: ", l_epoch_loss / len(val_dataloader))
print("KEYWORD LOSS: ", k_epoch_loss / len(val_dataloader))
